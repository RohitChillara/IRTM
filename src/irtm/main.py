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



stop_words = set(stopwords.words("english"))
documents = dict()
global_tokens = dict()

class Document:
    def __init__(self,document:list):
        self.doc_id = document[5]
        self.tweet = document[4]
        self.tweet_id = document[1]
        self.user_id = document[2]
        self.user_name = document[3]

    def getTweet(self):
        return [self.doc_id,self.tweet_id,self.user_id,self.user_name,self.tweet]


def normalize(tokens : list) ->list:
    for index,token in enumerate(tokens,start=0):
        tokens[index]=wnl().lemmatize(token)
    return tokens



def tokenize(text:str)->list:
    no_lines    = re.sub(r'\[(?:NEWLINE|TAB)\]', ' ', text)
    no_url      = re.sub(r'https?://\S+', ' ', no_lines)
    no_mentions = re.sub(r'(?<!\S)@\S+', ' ', no_url)

    # Letter runs, optionally joined by internal apostrophes (isn't stays one term).
    words = re.findall(r"[a-z]+(?:'[a-z]+)*", no_mentions.lower())

    return [w for w in words if w not in stop_words]


def read_csv(path:Path):
    fieldnames = ["cleaned"]
    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        reader = csv.DictReader(f,fieldnames=fieldnames,delimiter='\t',restkey="extra")
        for doc_id,row in enumerate(reader,start=1):
            columns = row["cleaned"].split('\t',maxsplit=4)
            columns.append(doc_id)
            if len(columns)<6:
                raise ValueError(f"Bad file received. Expected 5, received {len(columns)}")
            else:
                doc = Document(columns)
                documents[doc_id] = doc
                terms = set(normalize(tokenize(columns[4])))
                for term in



if __name__=="__main__":

    ensure_nltk("corpora/stopwords", "stopwords")
    ensure_nltk("corpora/wordnet", "wordnet")
    path = Path.home()/"ws26/IRTM/sample.csv"
    text = """RT @who_int: Malaria vaccine isn't risk-free… 😷💉[NEWLINE][NEWLINE]WHO’s new report on malaria—vaccines & side-effects: 100% safe?! #Malaria2018 #malaria
    Real line break here ðŸ‘€ðŸ‘‡ and “curly quotes” plus "straight quotes".
    Email me at a@b.com or visit https://t.co/aB3x (see also malaria_vaccine). malaria MALARIA Malaria."""

    tokens = tokenize(text)
    print(tokens)
    print(normalize(tokens))
