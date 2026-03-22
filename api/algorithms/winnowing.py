import hashlib

def get_hash(string: str) -> int:
    return int(hashlib.md5(string.encode('utf-8')).hexdigest()[:8], 16)

def winnow(text: str, k: int = 10, w: int = 5):
    # Remove spaces and newlines for fingerprinting
    text = text.lower().replace(" ", "").replace("\n", "")
    if len(text) < k:
        return set([(get_hash(text), 0)]) if text else set()
    
    hashes = []
    for i in range(len(text) - k + 1):
        kgram = text[i:i+k]
        hashes.append((get_hash(kgram), i))
    
    fingerprints = set()
    for i in range(len(hashes) - w + 1):
        window = hashes[i:i+w]
        min_hash = min(window, key=lambda x: x[0])
        fingerprints.add(min_hash)
    
    return fingerprints

def winnowing_similarity(text1: str, text2: str) -> float:
    """Returns Jaccard similarity of Winnowing fingerprints."""
    fp1 = winnow(text1)
    fp2 = winnow(text2)
    
    set1 = set([f[0] for f in fp1])
    set2 = set([f[0] for f in fp2])
    
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(intersection) / len(union)
