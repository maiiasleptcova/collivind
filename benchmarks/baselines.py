"""Plain baselines through the same data and scoring as collivind: dense cosine and BM25.

No engine, no dedup, no reranker: every LoCoMo turn / LongMemEval turn is one unit.

    uv run --extra embedded python benchmarks/baselines.py locomo benchmarks/data/locomo10.json OUT_DIR
    uv run --extra embedded python benchmarks/baselines.py lme benchmarks/data/longmemeval_s.json OUT_DIR

Writes ids-only ranking files (no dataset text) that benchmarks/compare.py scores.
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from locomo_bench import sessions_of, turn_text  # noqa: E402

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP = 50
STOP = set(
    "a an the and or of to in on at for with by is are was were be been do does did what when where who whom which "
    "how why that this it its as from has have had i you he she they we his her their my your our me him them "
    "not no so if but about into than then there would could should will can s t".split()
)


def toks(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        tf = [Counter(toks(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in tf], dtype=float)
        self.avg = self.len.mean() or 1.0
        df = Counter(w for c in tf for w in c)
        self.idf = {w: math.log(1 + (len(docs) - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.post = defaultdict(list)
        for i, c in enumerate(tf):
            for w, f in c.items():
                self.post[w].append((i, f))
        self.k1, self.b, self.n = k1, b, len(docs)

    def rank(self, query, top=TOP):
        scores = np.zeros(self.n)
        for w in set(toks(query)):
            for i, f in self.post.get(w, ()):
                norm = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                scores[i] += self.idf[w] * f * (self.k1 + 1) / norm
        return [int(i) for i in np.argsort(-scores)[:top] if scores[i] > 0]


def dense_ranker(model, docs):
    mat = model.encode(docs, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    def rank(query, top=TOP):
        q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        return [int(i) for i in np.argsort(-(mat @ q))[:top]]

    return rank


def locomo(dataset, out):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    rows = {"dense": [], "bm25": []}
    for sample in json.loads(Path(dataset).read_text()):
        turns = [(t["dia_id"], turn_text(t, when)) for _, when, ts in sessions_of(sample["conversation"]) for t in ts]
        ids, docs = [d for d, _ in turns], [x for _, x in turns]
        rankers = {"dense": dense_ranker(model, docs), "bm25": BM25(docs).rank}
        for i, qa in enumerate(sample["qa"]):
            for name, rank in rankers.items():
                units = [ids[j] for j in rank(qa["question"])]
                rows[name].append(
                    {"sample_id": sample["sample_id"], "qa_index": i, "category": qa["category"],
                     "ranked_units": [[u] for u in units], "unit_sessions": [u.split(":")[0] for u in units]}
                )  # fmt: skip
        print(sample["sample_id"], flush=True)
    for name, r in rows.items():
        Path(out, f"locomo_{name}.json").write_text(json.dumps(r))


def lme(dataset, out):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    rows = {"dense": [], "bm25": []}
    for n, q in enumerate(json.loads(Path(dataset).read_text()), 1):
        docs, sids, answer = [], [], []
        for sid, when, session in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
            for turn in session:
                content = (turn.get("content") or "").strip()
                if content:
                    docs.append(f"[{when}] {turn.get('role', 'user')}: {content}"[:2000])
                    sids.append(sid)
                    answer.append(bool(turn.get("has_answer")))
        for name, rank in {"dense": dense_ranker(model, docs), "bm25": BM25(docs).rank}.items():
            units = rank(q["question"])
            rows[name].append(
                {"question_id": q["question_id"], "question_type": q["question_type"],
                 "unit_sessions": [sids[j] for j in units], "unit_is_answer": [answer[j] for j in units]}
            )  # fmt: skip
        if n % 25 == 0:
            print(n, flush=True)
    for name, r in rows.items():
        Path(out, f"lme_{name}.json").write_text(json.dumps(r))


if __name__ == "__main__":
    Path(sys.argv[3]).mkdir(parents=True, exist_ok=True)
    {"locomo": locomo, "lme": lme}[sys.argv[1]](sys.argv[2], sys.argv[3])
