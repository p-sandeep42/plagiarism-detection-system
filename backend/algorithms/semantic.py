import re
import math
from collections import Counter


def _tokenize(text: str) -> list:
    """Lowercase and extract word tokens."""
    return re.findall(r'\w+', text.lower())


def _compute_tfidf(chunks: list) -> list:
    """Compute TF-IDF vectors for a list of text chunks.
    
    Returns a list of Counter dicts, one per chunk, with TF-IDF weights.
    """
    # Document frequency: how many chunks contain each term
    df = Counter()
    chunk_tokens = []
    for chunk in chunks:
        tokens = _tokenize(chunk)
        chunk_tokens.append(tokens)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            df[token] += 1

    n_docs = len(chunks)
    tfidf_vectors = []
    for tokens in chunk_tokens:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        tfidf = {}
        for term, count in tf.items():
            tf_val = count / total
            idf_val = math.log((n_docs + 1) / (df[term] + 1)) + 1  # smoothed IDF
            tfidf[term] = tf_val * idf_val
        tfidf_vectors.append(tfidf)

    return tfidf_vectors


def _cosine_sim(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between two sparse vectors (dicts)."""
    if not vec_a or not vec_b:
        return 0.0

    # Dot product
    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common_keys)

    # Magnitudes
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def cosine_similarity_matrix(vecs_a: list, vecs_b: list) -> list:
    """Compute pairwise cosine similarity matrix between two lists of vectors.
    
    Returns a 2D list where result[i][j] = cosine_sim(vecs_a[i], vecs_b[j]).
    """
    matrix = []
    for va in vecs_a:
        row = [_cosine_sim(va, vb) for vb in vecs_b]
        matrix.append(row)
    return matrix


def semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts using TF-IDF + cosine similarity."""
    if not text1 or not text2:
        return 0.0

    chunk_size = 500
    chunks1 = [text1[i:i+chunk_size] for i in range(0, len(text1), chunk_size)]
    chunks2 = [text2[i:i+chunk_size] for i in range(0, len(text2), chunk_size)]

    if not chunks1 or not chunks2:
        return 0.0

    # Limit to first 20 chunks to speed up processing
    chunks1 = chunks1[:20]
    chunks2 = chunks2[:20]

    # Build a shared vocabulary for IDF by combining all chunks
    all_chunks = chunks1 + chunks2
    df = Counter()
    all_token_lists = []
    for chunk in all_chunks:
        tokens = _tokenize(chunk)
        all_token_lists.append(tokens)
        for token in set(tokens):
            df[token] += 1

    n_docs = len(all_chunks)

    # Compute TF-IDF for each chunk using the shared IDF
    def _tfidf_with_shared_idf(tokens):
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        tfidf = {}
        for term, count in tf.items():
            tf_val = count / total
            idf_val = math.log((n_docs + 1) / (df[term] + 1)) + 1
            tfidf[term] = tf_val * idf_val
        return tfidf

    vecs1 = [_tfidf_with_shared_idf(all_token_lists[i]) for i in range(len(chunks1))]
    vecs2 = [_tfidf_with_shared_idf(all_token_lists[len(chunks1) + i]) for i in range(len(chunks2))]

    # Compute similarity matrix
    sim_matrix = cosine_similarity_matrix(vecs1, vecs2)

    # Max pool: for each chunk in doc1, find the most similar chunk in doc2
    max_scores = [max(row) if row else 0.0 for row in sim_matrix]

    # Average these max scores
    score = sum(max_scores) / len(max_scores) if max_scores else 0.0

    # Clamp between 0 and 1
    return max(0.0, min(1.0, score))


def encode_sentences(sentences: list) -> list:
    """Encode sentences as TF-IDF vectors using shared IDF.
    
    This replaces MODEL.encode() for use in semantic highlighting.
    Returns a list of TF-IDF vector dicts.
    """
    if not sentences:
        return []

    df = Counter()
    token_lists = []
    for sent in sentences:
        tokens = _tokenize(sent)
        token_lists.append(tokens)
        for token in set(tokens):
            df[token] += 1

    n_docs = len(sentences)
    vectors = []
    for tokens in token_lists:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        tfidf = {}
        for term, count in tf.items():
            tf_val = count / total
            idf_val = math.log((n_docs + 1) / (df[term] + 1)) + 1
            tfidf[term] = tf_val * idf_val
        vectors.append(tfidf)

    return vectors
