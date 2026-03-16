import sys
import re
sys.path.append('backend')
from algorithms.semantic import model as sem_model
from sklearn.metrics.pairwise import cosine_similarity

text1 = open('test_doc1.txt').read()
text2 = open('test_doc2.txt').read()

sentences1 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text1) if len(s.strip()) > 10]
sentences2 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text2) if len(s.strip()) > 10]

print(f"Sentences1: {len(sentences1)}")
print(f"Sentences2: {len(sentences2)}")

if sentences1 and sentences2 and sem_model:
    emb1 = sem_model.encode(sentences1)
    emb2 = sem_model.encode(sentences2)
    sim_matrix = cosine_similarity(emb1, emb2)
    print("Sim matrix shape:", sim_matrix.shape)
    for i, s1 in enumerate(sentences1):
        best_j = sim_matrix[i].argmax()
        best_sim = sim_matrix[i][best_j]
        print(f"Index {i} -> max sim {best_sim} with index {best_j}")
else:
    print("sem_model is missing!")
