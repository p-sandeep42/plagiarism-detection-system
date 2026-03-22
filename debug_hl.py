import sys
import re
sys.path.append('backend')
from algorithms.semantic import encode_sentences, cosine_similarity_matrix

text1 = open('test_doc1.txt').read()
text2 = open('test_doc2.txt').read()

sentences1 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text1) if len(s.strip()) > 10]
sentences2 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text2) if len(s.strip()) > 10]

print(f"Sentences1: {len(sentences1)}")
print(f"Sentences2: {len(sentences2)}")

if sentences1 and sentences2:
    all_sentences = sentences1 + sentences2
    all_vecs = encode_sentences(all_sentences)
    emb1 = all_vecs[:len(sentences1)]
    emb2 = all_vecs[len(sentences1):]
    sim_matrix = cosine_similarity_matrix(emb1, emb2)
    print("Sim matrix size:", len(sim_matrix), "x", len(sim_matrix[0]) if sim_matrix else 0)
    for i, s1 in enumerate(sentences1):
        best_j = max(range(len(sim_matrix[i])), key=lambda j: sim_matrix[i][j])
        best_sim = sim_matrix[i][best_j]
        print(f"Index {i} -> max sim {best_sim:.4f} with index {best_j}")
else:
    print("No sentences found!")
