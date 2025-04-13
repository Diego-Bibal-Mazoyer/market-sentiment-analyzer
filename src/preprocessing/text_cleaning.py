import re
import nltk
import spacy
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)                    # URLs
    text = re.sub(r"[^a-z\s]", "", text)                   # Punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip()               # Extra whitespace
    return text

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in STOPWORDS])

def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.lemma_ != "-PRON-"])

def preprocess_pipeline(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text

