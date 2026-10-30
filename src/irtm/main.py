from __future__ import annotations
import re
from nltk.stem import WordNetLemmatizer as wnl
from nltk.corpus import stopwords


stop_words = set(stopwords.words("english"))

def normalize(tokens : list) ->list:
    normalized = list()
    for token in tokens:
        normalized.append(wnl().lemmatize(token))



def tokenize(text:str)->list:
    no_lines    = re.sub(r'\[(?:NEWLINE|TAB)\]', ' ', text)
    no_url      = re.sub(r'https?://\S+', ' ', no_lines)
    no_mentions = re.sub(r'(?<!\S)@\S+', ' ', no_url)

    # Letter runs, optionally joined by internal apostrophes (isn't stays one term).
    words = re.findall(r"[a-z]+(?:'[a-z]+)*", no_mentions.lower())

    return [w for w in words if w not in stop_words]


