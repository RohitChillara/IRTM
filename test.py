import nltk
from nltk.stem import WordNetLemmatizer

# One-time download, skipped if the data is already installed
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()

for t in ["following", "suffering", "meeting", "saw", "working", "vaccinated"]:
    as_verb = lemmatizer.lemmatize(t, "v")
    both = lemmatizer.lemmatize(as_verb, "n")
    print(f"{t:12} verb: {as_verb:12} verb+noun: {both}")