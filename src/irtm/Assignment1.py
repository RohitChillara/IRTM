"""IRTM Assignment 1 - Programming Task 1: non-positional inverted index over tweets.

Group members: Rohit Chillara, Shrery Parmar, Manmeet Kohli
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from itertools import islice
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk(resource, name):
    """Download an NLTK resource if it is not installed yet."""
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(name, quiet=True)


# Must run before the stopword list is loaded below.
ensure_nltk("corpora/stopwords", "stopwords")
ensure_nltk("corpora/wordnet", "wordnet")

# Contractions that stay whole after tokenizing (the 's forms are handled in tokenize()).
CONTRACTIONS = {
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
    "haven't", "hasn't", "hadn't", "shouldn't", "can't", "won't", "ain't",
    "i'm", "you're", "we're", "they're",
    "i've", "you've", "we've", "they've", "should've",
    "i'll", "you'll", "he'll", "she'll", "it'll", "we'll", "they'll", "that'll",
    "i'd", "you'd", "he'd", "she'd", "we'd", "they'd",
    "cannot",
}

stop_words = set(stopwords.words("english"))
STOP = stop_words | CONTRACTIONS

documents = dict()       # docID -> Document (tweet data, kept outside the index)
dictionary = dict()      # term -> {"term": ..., "size": ..., "pointer": postings ID}
postings_store = dict()  # postings ID -> PostingsList (separate from the dictionary)


class Document:
    """One tweet and its metadata."""

    def __init__(self, document: dict):
        self.doc_id = document["docID"]
        self.tweet = document["tweet"]
        self.tweet_id = document["tweetID"]
        self.user_id = document["userID"]
        self.user_name = document["userName"]

    def getTweet(self):
        return [self.doc_id, self.tweet_id, self.user_id, self.user_name, self.tweet]


class PostingsList:
    """Sorted list of docIDs for one term; the list position acts as the 'next' pointer."""

    def __init__(self):
        self._ids: list[int] = []

    def add(self, doc_id: int) -> None:
        """Append a docID; docIDs must arrive in ascending order, repeats are ignored."""
        if self._ids:
            if self._ids[-1] == doc_id:
                return
            if self._ids[-1] > doc_id:
                raise ValueError(f"docID {doc_id} after {self._ids[-1]}")
        self._ids.append(doc_id)

    def __len__(self):
        return len(self._ids)

    def __iter__(self):
        return iter(self._ids)          # the intersection must use iterators


_lemmatizer = WordNetLemmatizer()       # created once, not per token


@lru_cache(maxsize=None)
def lemma(word: str) -> str:
    """Lemmatize one word (noun, verb, noun); cached because the vocabulary is small."""
    return _lemmatizer.lemmatize(_lemmatizer.lemmatize(_lemmatizer.lemmatize(word, "n"), "v"), "n")


def normalize(tokens: list) -> list:
    """Lemmatize every token and return a new list."""
    return [lemma(t) for t in tokens]


def tokenize(text: str) -> list:
    """Clean a tweet and return its lowercase word tokens without stopwords."""
    no_lines = re.sub(r'\[(?:NEWLINE|TAB)\]', ' ', text)
    no_url = re.sub(r'https?://\S+', ' ', no_lines)
    no_mentions = re.sub(r'(?<!\S)@\S+', ' ', no_url)

    lowered = no_mentions.lower().replace("’", "'")      # curly -> straight apostrophe
    no_poss = re.sub(r"'s(?![a-z])", " ", lowered)       # drop possessive / 's contraction

    # Letter runs, optionally joined by internal apostrophes (isn't stays one term).
    words = re.findall(r"[a-z]+(?:'[a-z]+)*", no_poss)

    return [w for w in words if w not in STOP]


def add_posting(term: str, doc_id: int) -> None:
    """Add doc_id to the postings of term and keep the dictionary entry in sync."""
    entry = dictionary.get(term)
    if entry is None:
        pid = len(postings_store)                        # next free postings ID
        postings_store[pid] = PostingsList()
        entry = dictionary[term] = {"term": term, "size": 0, "pointer": pid}
    postings = postings_store[entry["pointer"]]
    postings.add(doc_id)
    entry["size"] = len(postings)


def index(filename) -> None:
    """Read the tweet file and build the dictionary, postings lists and document store."""
    fieldnames = ["timeStamp", "tweetID", "userID", "userName", "tweet"]
    seen = set()
    rows_read = 0
    dup_rows = 0
    bad_rows = 0

    with open(filename, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, fieldnames=fieldnames, delimiter='\t', restkey="extra")
        for doc_id, row in enumerate(reader, start=1):
            rows_read = doc_id

            tid = row["tweetID"]
            text = row["tweet"]
            if not tid or not text or not text.strip():     # missing fields: discard
                bad_rows += 1
                continue

            if tid in seen:                                  # repeated tweet ID: keep the first
                dup_rows += 1
                continue
            seen.add(tid)

            row["docID"] = doc_id
            documents[doc_id] = Document(row)

            terms = set(normalize(tokenize(text)))           # set: one posting per tweet and term
            for term in terms:
                add_posting(term, doc_id)

    print(f"rows read: {rows_read} | indexed: {len(documents)} | "
          f"duplicate IDs skipped: {dup_rows} | bad rows skipped: {bad_rows}")


def _term_iter(raw: str):
    """Iterator over the postings of one query term, or None if the term yields nothing usable."""
    toks = normalize(tokenize(raw))
    if not toks:                                  # empty after cleaning, or all stopwords
        return None
    term = toks[0]                                # a term like "side-effects" yields several; take the first
    entry = dictionary.get(term)
    if entry is None:                             # never indexed
        return None
    return iter(postings_store[entry["pointer"]])


def query(term1: str, term2: str = None) -> list:
    """query(term): docIDs of one term. query(term1, term2): docIDs containing both (logical and)."""
    a = _term_iter(term1)
    if term2 is None:
        return [] if a is None else list(a)
    b = _term_iter(term2)
    if a is None or b is None:                    # AND: a missing term means no results
        return []
    return list(intersect(a, b))


def query_all(text: str) -> list:
    """AND over all terms of a free-text query; terms missing from the index are skipped."""
    tokens = set(normalize(tokenize(text)))
    doc_list = [dictionary[t] for t in tokens if t in dictionary]
    doc_list.sort(key=lambda e: e["size"])        # shortest postings list first
    iters = [iter(postings_store[e["pointer"]]) for e in doc_list]
    if not iters:                                 # no usable terms: empty result
        return []

    result = iters[0]
    for it in iters[1:]:
        result = intersect(result, it)
    return list(result)


def intersect(a, b):
    """Merge-intersect two iterators over sorted docIDs (no array-style access)."""
    x = next(a, None)
    y = next(b, None)
    while x is not None and y is not None:
        if x == y:
            yield x
            x = next(a, None)
            y = next(b, None)
        elif x < y:
            x = next(a, None)
        else:
            y = next(b, None)


if __name__ == "__main__":
    index(Path.home() / "ws26/IRTM/tweets.csv")

    # Information need: side effects of malaria vaccines.
    runs = [
        ("query('malaria', 'vaccine')", query("malaria", "vaccine")),
        ("query_all('side effects of malaria vaccines')", query_all("side effects of malaria vaccines")),
    ]
    for label, hits in runs:
        print(label, "->", len(hits), "tweets")
        for doc_id in islice(hits, 10):
            d = documents[doc_id]
            print(d.tweet_id, d.tweet[:120])