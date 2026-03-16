import difflib
import re
import sys
sys.path.append('backend')
from algorithms.semantic import get_model
from sklearn.metrics.pairwise import cosine_similarity

text1 = open('test_doc1.txt').read()
text2 = open('test_doc2.txt').read()

matcher = difflib.SequenceMatcher(None, text1, text2)
blocks = matcher.get_matching_blocks()
print("Exact match blocks > 5:")
for b in blocks:
    if b.size > 5:
        print(f"Match: {text1[b.a:b.a+b.size]}")

sentences1 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text1) if len(s.strip()) > 10]
sentences2 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text2) if len(s.strip()) > 10]
print("Sentences 1:", len(sentences1))
print("Sentences 2:", len(sentences2))

model = get_model()
emb1 = model.encode(sentences1)
emb2 = model.encode(sentences2)
sim = cosine_similarity(emb1, emb2)
print("Similarity matrix:\n", sim)
