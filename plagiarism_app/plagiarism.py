
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text: str) -> str:
    # basic cleaning: lowercase, remove extra spaces, keep words/numbers
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_similarity(text1: str, text2: str) -> float:
    t1 = preprocess(text1)
    t2 = preprocess(text2)

    if not t1 or not t2:
        return 0.0

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([t1, t2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return float(score)
