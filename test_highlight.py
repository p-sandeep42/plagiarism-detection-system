import difflib
import re
import sys
sys.path.append('backend')
from algorithms.semantic import encode_sentences, cosine_similarity_matrix

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

all_sentences = sentences1 + sentences2
all_vecs = encode_sentences(all_sentences)
emb1 = all_vecs[:len(sentences1)]
emb2 = all_vecs[len(sentences1):]
sim = cosine_similarity_matrix(emb1, emb2)
print("Similarity matrix:")
for row in sim:
    print([f"{v:.4f}" for v in row])
