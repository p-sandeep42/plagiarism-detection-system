from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ CREATE APP FIRST
app = Flask(__name__)

# =========================
# Routes
# =========================

@app.route("/")
def index():
    return render_template("index.html")


def compute_similarity(text1: str, text2: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return score * 100


@app.route("/check", methods=["POST"])
def check():
    f1 = request.files.get("file1")
    f2 = request.files.get("file2")

    if not f1 or not f2 or f1.filename == "" or f2.filename == "":
        return render_template("index.html", error="Please upload BOTH files."), 200

    try:
        text1 = f1.read().decode("utf-8", errors="ignore")
        text2 = f2.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return render_template("index.html", error=f"File read error: {e}"), 200

    similarity_score = compute_similarity(text1, text2)
    similarity_display = round(similarity_score, 2)

    if similarity_score < 15:
        label = "Low Similarity"
        suggestion = "Looks good. You can still improve by rewriting common phrases."
    elif similarity_score < 30:
        label = "Moderate Similarity"
        suggestion = "Try paraphrasing repeated sentences and remove common definitions."
    else:
        label = "High Similarity"
        suggestion = "Rewrite copied lines, remove repeated definitions, and avoid common templates."

    return render_template(
        "result.html",
        similarity=similarity_display,
        label=label,
        suggestion=suggestion,
        highlights=None
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)