import torch
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    model = None
except Exception:
    model = None

def semantic_similarity(text1: str, text2: str) -> float:
    if model is None:
        return 0.0
        
    chunk_size = 500
    chunks1 = [text1[i:i+chunk_size] for i in range(0, len(text1), chunk_size)]
    chunks2 = [text2[i:i+chunk_size] for i in range(0, len(text2), chunk_size)]
    
    if not chunks1 or not chunks2:
        return 0.0

    # Limit to first 20 chunks to speed up processing
    chunks1 = chunks1[:20]
    chunks2 = chunks2[:20]
    
    # Get embeddings
    with torch.no_grad():
        emb1 = model.encode(chunks1, convert_to_tensor=True)
        emb2 = model.encode(chunks2, convert_to_tensor=True)
        
        # Compute cosine similarities
        cosine_scores = util.cos_sim(emb1, emb2)
        
        # Max pool: for each chunk in doc1, what is the most similar chunk in doc2?
        max_scores, _ = torch.max(cosine_scores, dim=1)
        
        # Average these max scores for an overall score
        score = float(torch.mean(max_scores).item())
        
        # Make sure score is between 0 and 1
        return max(0.0, min(1.0, score))
