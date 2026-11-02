from __future__ import annotations
import re
from nltk.stem import WordNetLemmatizer as wnl
from nltk.corpus import stopwords
from pathlib import Path
import csv
import nltk

def ensure_nltk(resource, name):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(name, quiet=True)

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
# stop_words_de = set(stopwords.words("german"))

STOP = stop_words | CONTRACTIONS
documents = dict()
global_tokens = dict()

class Document:
    def __init__(self,document:dict):
        self.doc_id = document["docID"]
        self.tweet = document["tweet"]
        self.tweet_id = document["tweetID"]
        self.user_id = document["userID"]
        self.user_name = document["userName"]

    def getTweet(self):
        return [self.doc_id,self.tweet_id,self.user_id,self.user_name,self.tweet]

class PostingsList:
    def __init__(self):
        self._ids: list[int] = []

    def add(self, doc_id: int) -> None:
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


def normalize(tokens : list) ->list:
    for index,token in enumerate(tokens,start=0):
        tokens[index]=wnl().lemmatize(wnl().lemmatize(wnl().lemmatize(token,"n"),"v"),"n")
    return tokens



def tokenize(text: str) -> list:
    no_lines    = re.sub(r'\[(?:NEWLINE|TAB)\]', ' ', text)
    no_url      = re.sub(r'https?://\S+', ' ', no_lines)
    no_mentions = re.sub(r'(?<!\S)@\S+', ' ', no_url)

    lowered = no_mentions.lower().replace("’", "'")      # curly -> straight apostrophe
    no_poss = re.sub(r"'s(?![a-z])", " ", lowered)       # drop possessive / 's contraction

    # Letter runs, optionally joined by internal apostrophes (isn't stays one term).
    words = re.findall(r"[a-z]+(?:'[a-z]+)*", no_poss)

    return [w for w in words if w not in STOP]


def read_csv(path:Path):
    fieldnames = ["timeStamp","tweetID","userID","userName","tweet"]
    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        reader = csv.DictReader(f,fieldnames=fieldnames,delimiter='\t',restkey="extra")
        for doc_id,row in enumerate(reader,start=1): 
            row["docID"] = doc_id
            doc = Document(row)
            documents[doc_id] = doc
            terms = set(normalize(tokenize(row["tweet"])))
            for term in terms:
                if term in global_tokens:
                    global_tokens[term]["pointer"].add(doc_id)
                    global_tokens[term]["size"] = global_tokens[term]["pointer"].__len__()
                else:
                    postings = PostingsList()
                    postings.add(doc_id=doc_id)
                    global_tokens[term] = {"size":1,"pointer":postings}



def query(terms: str)->list:
    tokens = set(normalize(tokenize(terms)))
    doc_list = list()
    for term in tokens:
        if term in global_tokens:
            doc_list.append(global_tokens[term])
        else:
            continue
    doc_list.sort(key=lambda e: e["size"])
    iters = [iter(e["pointer"]) for e in doc_list]
    if not iters:
        return iter(())              # no usable terms: empty result

    result = iters[0]
    for it in iters[1:]:
        result = intersect(result, it)
    return result


def intersect(a,b):
    x = next(a,None)
    y = next(b,None)
    while x is not None and y is not None:
        if x == y:
            yield x
        elif x > y:
            y = next(b,None)

        elif y > x:
            x = next(a,None)

if __name__=="__main__":

    ensure_nltk("corpora/stopwords", "stopwords")
    ensure_nltk("corpora/wordnet", "wordnet")
    path = Path.home()/"ws26/IRTM/tweets.csv"
    read_csv(path)

    hits = query("side effects of vaccines")
    docs = [documents[doc_id] for doc_id in hits]
    for d in docs[:10]:
        print(d.tweet_id, d.tweet[:120])
