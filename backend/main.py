from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import itertools
import asyncio
import os
import sys
import concurrent.futures
import difflib
import re

# Ensure backend directory is in the import path for Vercel serverless functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.schemas import ComparisonResponse, ComparisonScore, HighlightInfo, BatchComparisonResponse, StudentMeta, PairScore, StudentSummary
from services.parser import parse_file

# Import algorithms
from algorithms.winnowing import winnowing_similarity
from algorithms.structural import structural_similarity
from algorithms.semantic import semantic_similarity, encode_sentences, cosine_similarity_matrix

app = FastAPI(title="AuraDiff API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow frontend to access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
session_queues = {}

@app.get("/")
def health_check():
    return {"status": "ok", "message": "AuraDiff Backend is running"}

def _get_highlights(text1: str, text2: str) -> List[HighlightInfo]:
    matcher = difflib.SequenceMatcher(None, text1, text2)
    matching_blocks = matcher.get_matching_blocks()
    
    highlights = []
    
    # 1. Exact Matches (Winnowing / Difflib)
    for block in matching_blocks:
        if block.size > 5: # Filter out very short noisy matches
            highlights.append(
                HighlightInfo(
                    source_index_start=block.a,
                    source_index_end=block.a + block.size,
                    target_index_start=block.b,
                    target_index_end=block.b + block.size,
                    text=text1[block.a:block.a + block.size],
                    match_type="exact"
                )
            )
            
    # 2. Semantic Matches (Sentence Level)
    try:
        sentences1 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text1) if len(s.strip()) > 10]
        sentences2 = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text2) if len(s.strip()) > 10]
        
        if sentences1 and sentences2:
            all_sentences = sentences1 + sentences2
            all_vecs = encode_sentences(all_sentences)
            emb1 = all_vecs[:len(sentences1)]
            emb2 = all_vecs[len(sentences1):]
            sim_matrix = cosine_similarity_matrix(emb1, emb2)
            
            for i, s1 in enumerate(sentences1):
                best_j = max(range(len(sim_matrix[i])), key=lambda j: sim_matrix[i][j])
                best_sim = sim_matrix[i][best_j]
                
                if best_sim > 0.50: # High semantic similarity threshold
                    s2 = sentences2[best_j]
                    start1 = text1.find(s1)
                    start2 = text2.find(s2)
                    
                    if start1 != -1 and start2 != -1:
                        # Ensure it doesn't overlap excessively with an exact match
                        is_covered = any(
                            (h.source_index_start <= start1 and h.source_index_end >= start1 + len(s1)) 
                            for h in highlights
                        )
                        if not is_covered:
                            highlights.append(
                                HighlightInfo(
                                    source_index_start=start1,
                                    source_index_end=start1 + len(s1),
                                    target_index_start=start2,
                                    target_index_end=start2 + len(s2),
                                    text=s1,
                                    match_type="semantic"
                                )
                            )
    except Exception as e:
        print(f"Error in semantic highlighting: {e}")
        
    return highlights

def _aggregate(w_score: float, s_score: float, e_score: float, ext1: str) -> float:
    if ext1 in ["py", "js", "cpp", "java", "ts"]:
        w_win = 0.2
        w_str = 0.6
        w_sem = 0.2
    else:
        w_win = 0.3
        w_str = 0.2
        w_sem = 0.5
    return (w_win * w_score) + (w_str * s_score) + (w_sem * e_score)

@app.post("/compare", response_model=ComparisonResponse)
async def compare_files(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    if not file1.filename or not file2.filename:
        raise HTTPException(status_code=400, detail="Filenames are required")

    ext1 = file1.filename.split(".")[-1].lower()
    ext2 = file2.filename.split(".")[-1].lower()
    
    content1_bytes = await file1.read()
    content2_bytes = await file2.read()
    
    text1 = parse_file(file1.filename, content1_bytes)
    text2 = parse_file(file2.filename, content2_bytes)
    
    winnowing_score = winnowing_similarity(text1, text2)
    struct_score = structural_similarity(text1, text2, ext1 if ext1 == ext2 else "txt")
    sem_score = semantic_similarity(text1, text2)
    
    total_score = _aggregate(winnowing_score, struct_score, sem_score, ext1 if ext1 == ext2 else "txt")
    highlights = _get_highlights(text1, text2)
    
    score_obj = ComparisonScore(
        winnowing_score=round(winnowing_score, 4),
        jaccard_score=round(struct_score, 4),
        semantic_score=round(sem_score, 4),
        total_score=round(total_score, 4)
    )
    
    return ComparisonResponse(
        scores=score_obj,
        highlights=highlights,
        message="Comparison completed successfully.",
        source_text=text1,
        target_text=text2
    )

def _name_from_filename(filename: str) -> str:
    name, _ = os.path.splitext(filename)
    return name.replace("_", " ").replace("-", " ").title()

def _validate_session_token(token: str) -> bool:
    secret = os.getenv('BATCH_WS_SECRET', 'replace_with_random_32_char_string')
    return token == secret

def _build_summary(student_index: int, matrix: List[List[Optional[PairScore]]], n: int) -> StudentSummary:
    scores = [matrix[student_index][j].total_score for j in range(n) if student_index != j and matrix[student_index][j] is not None]
    max_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    if max_score < 0.30:
        risk = "Low"
    elif max_score <= 0.60:
        risk = "Medium"
    elif max_score <= 0.79:
        risk = "High"
    else:
        risk = "Critical"
        
    return StudentSummary(
        student_index=student_index,
        max_score=max_score,
        avg_score=avg_score,
        risk_level=risk
    )

def _run_pair_sync(text_a, text_b, type_a, type_b):
    w = winnowing_similarity(text_a, text_b)
    s = structural_similarity(text_a, text_b, type_a if type_a == type_b else "txt")
    e = semantic_similarity(text_a, text_b)
    total = _aggregate(w, s, e, type_a if type_a == type_b else "txt")
    highlights = _get_highlights(text_a, text_b)
    
    return PairScore(
        winnowing_score=round(w, 4),
        jaccard_score=round(s, 4),
        semantic_score=round(e, 4),
        total_score=round(total, 4),
        highlights=highlights
    )

@app.post("/compare-batch", response_model=BatchComparisonResponse)
async def compare_batch(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None)
):
    n = len(files)
    max_files = int(os.getenv('MAX_BATCH_FILES', 30))
    if n < 2 or n > max_files:
        raise HTTPException(400, detail=f"Requires 2–{max_files} files")
        
    parsed = []
    texts = []
    
    for f in files:
        ext = f.filename.split(".")[-1].lower() if f.filename else "txt"
        content = await f.read()
        text = parse_file(f.filename, content)
        parsed.append((text, ext))
        texts.append(text)
        
    students = [
        StudentMeta(
            student_index=i,
            name=_name_from_filename(files[i].filename),
            filename=files[i].filename
        ) for i in range(n)
    ]
    
    pairs = list(itertools.combinations(range(n), 2))
    total_pairs = len(pairs)
    matrix = [[None]*n for _ in range(n)]
    
    completed = 0
    queue = session_queues.get(session_id) if session_id else None
        
    async def process_pair(i, j, text_a, text_b, type_a, type_b):
        nonlocal completed
        loop = asyncio.get_event_loop()
        try:
            res = await loop.run_in_executor(pool, _run_pair_sync, text_a, text_b, type_a, type_b)
        except Exception as e:
            print(f"Error in pair {i}, {j}: {e}")
            res = PairScore(winnowing_score=0, jaccard_score=0, semantic_score=0, total_score=0, highlights=[])
            
        completed += 1
        if queue:
            await queue.put({
                "completed": completed,
                "total": total_pairs,
                "current_pair": [i, j]
            })
        return i, j, res

    tasks = [process_pair(i, j, parsed[i][0], parsed[j][0], parsed[i][1], parsed[j][1]) for i, j in pairs]
    results = await asyncio.gather(*tasks)
    
    for i, j, res in results:
        matrix[i][j] = res
        matrix[j][i] = res
        
    summary = sorted([_build_summary(i, matrix, n) for i in range(n)], key=lambda s: s.max_score, reverse=True)
    
    if queue:
        await queue.put("DONE")
    
    return BatchComparisonResponse(
        students=students,
        matrix=matrix,
        summary=summary,
        parsed_texts=texts
    )

@app.websocket("/ws/batch-progress")
async def batch_progress_ws(ws: WebSocket, token: str = Query(...)):
    if not _validate_session_token(token):
        await ws.close(code=1008)
        return
    await ws.accept()
    
    session_id = None
    try:
        data = await ws.receive_json()
        session_id = data.get("session_id")
        
        if session_id:
            queue = asyncio.Queue()
            session_queues[session_id] = queue
            
            while True:
                msg = await queue.get()
                if msg == "DONE":
                    break
                await ws.send_json(msg)
    except Exception as e:
        print(f"WS Error: {e}")
    finally:
        if session_id and session_id in session_queues:
            del session_queues[session_id]
