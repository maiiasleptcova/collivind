"""Query terms for the keyword (BM25) channel, shared by the SQLite and Neo4j stores.

Both backends get the same input: lowercase alphanumeric words, English
stopwords dropped, deduplicated, capped. Each term is quoted/escaped by the
store and the terms are OR-ed, so punctuation in a coding prompt can never
reach the full-text query parser as syntax.
"""

import re
from typing import List

# ponytail: first MAX_TERMS distinct words of the query. A multi-KB prompt
# keeps only its opening; pick terms by IDF if long prompts lose recall.
MAX_TERMS = 32

STOPWORDS = frozenset(
    "a an the and or of to in on at for with by is are was were be been do does did what when where who whom which "
    "how why that this it its as from has have had i you he she they we his her their my your our me him them "
    "not no so if but about into than then there would could should will can s t".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def query_terms(text: str, max_terms: int = MAX_TERMS) -> List[str]:
    terms: List[str] = []
    for word in _WORD.findall(text.lower()):
        if word not in STOPWORDS and word not in terms:
            terms.append(word)
            if len(terms) == max_terms:
                break
    return terms
