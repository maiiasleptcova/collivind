// What the rail says above its matches, as plain data so node can test it
// (tests/unit/test_web_search_hint.py). No DOM here.
//
// The order matters: a search by meaning still in flight outranks the live
// text filter. On a cold start the first search waits seconds for the
// embedding model, and the filter used to say "Nothing matches. Press Enter
// to search by meaning" meanwhile, so readers pressed Enter a second time.

export function railHint({ search, searching, text }) {
  if (searching) {
    return { heading: `Searching by meaning for “${searching}”…`, empty: "The first search after a start can take a few seconds while the model loads." };
  }
  if (search) return { heading: `Search results for “${search.query}”`, empty: "Nothing matches. Try other words, or clear the project filter." };
  if (text) return { heading: `Matching “${text}”`, empty: "Nothing matches. Press Enter to search by meaning instead." };
  return null;
}
