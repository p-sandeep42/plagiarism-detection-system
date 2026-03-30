"""
AuraDiff API — Main application
Includes: auth, history, CORS, rate limiting, input validation, global error handling.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, Query, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import itertools
import asyncio
import os
import sys
import json
import concurrent.futures
import difflib
import re
import traceback
import html

# Ensure backend directory is in the import path for Vercel serverless functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.schemas import (
    ComparisonResponse, ComparisonScore, HighlightInfo,
    BatchComparisonResponse, StudentMeta, PairScore, StudentSummary,
)
from services.parser import parse_file
from algorithms.winnowing import winnowing_similarity
from algorithms.structural import structural_similarity
# Semantic removed
# Auth and History imports removed

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="AuraDiff API")

# Origins allow-list reused for both HTTP CORS and WebSocket origin check
ALLOWED_ORIGINS: list[str] = []  # populated after env read below

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow configured origins (defaults to permissive for dev)
_raw_origins = os.getenv("CORS_ORIGINS", "*").split(",")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth and Database initialization removed

# Global variables
pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
session_queues: Dict[str, asyncio.Queue] = {}

# ---------------------------------------------------------------------------
# Constants for validation
# ---------------------------------------------------------------------------
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"txt", "py", "js", "ts", "java", "cpp", "c", "h", "md", "csv", "json", "pdf", "docx"}
MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", "30"))

# ---------------------------------------------------------------------------
# Global exception handler — never leak stack traces
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log the full error server-side
    traceback.print_exc()
    # Return sanitized message to client
    return JSONResponse(
        status_code=500,
        content={"error": "An internal server error occurred. Please try again later."},
    )

# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def _sanitize_filename(filename: str) -> str:
    """Strip path components and HTML/script from filename."""
    if not filename:
        return "unnamed"
    # Remove path separators
    name = filename.replace("\\", "/").split("/")[-1]
    # Strip HTML tags
    name = re.sub(r"<[^>]+>", "", name)
    # Escape HTML entities
    name = html.escape(name)
    # Limit length
    return name[:255]


def _validate_file(f: UploadFile) -> str:
    """Validate a single file upload. Returns sanitized extension."""
    if not f.filename:
        raise HTTPException(400, detail="Filename is required")

    safe_name = _sanitize_filename(f.filename)
    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            detail=f"File type '.{ext}' is not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return ext


async def _read_validated(f: UploadFile) -> bytes:
    """Read file content with size validation."""
    content: bytes = await f.read()  # type: ignore[assignment]
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, detail=f"File '{f.filename}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
    if len(content) == 0:
        raise HTTPException(400, detail=f"File '{f.filename}' is empty")
    return content


# ---------------------------------------------------------------------------
# Health check (public)
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "AuraDiff Backend is running"}


# ---------------------------------------------------------------------------
# Highlighting logic
# ---------------------------------------------------------------------------
def _get_highlights(text1: str, text2: str) -> List[HighlightInfo]:
    matcher = difflib.SequenceMatcher(None, text1, text2)
    matching_blocks = matcher.get_matching_blocks()

    highlights = []

    # 1. Exact Matches
    for block in matching_blocks:
        if block.size > 5:
            highlights.append(
                HighlightInfo(
                    source_index_start=block.a,
                    source_index_end=block.a + block.size,
                    target_index_start=block.b,
                    target_index_end=block.b + block.size,
                    text=text1[block.a : block.a + block.size],
                    match_type="exact",
                )
            )

    return highlights


def _aggregate(w_score: float, s_score: float, e_score: float, ext1: str) -> float:
    # Semantic score is ignored to save memory usage on Render Free Tier. Weighting redistributed.
    if ext1 in ["py", "js", "cpp", "java", "ts"]:
        w_win, w_str = 0.3, 0.7
    else:
        w_win, w_str = 0.6, 0.4
    return (w_win * w_score) + (w_str * s_score)


# ---------------------------------------------------------------------------
# Pairwise compare (protected + rate limited)
# ---------------------------------------------------------------------------
@app.post("/compare", response_model=ComparisonResponse)
@limiter.limit("10/minute")
async def compare_files(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
):
    ext1 = _validate_file(file1)
    ext2 = _validate_file(file2)

    content1 = await _read_validated(file1)
    content2 = await _read_validated(file2)

    text1 = parse_file(file1.filename, content1)
    text2 = parse_file(file2.filename, content2)

    winnowing_score = winnowing_similarity(text1, text2)
    struct_score = structural_similarity(text1, text2, ext1 if ext1 == ext2 else "txt")
    sem_score = 0.0  # Semantic removed

    total_score = _aggregate(winnowing_score, struct_score, sem_score, ext1 if ext1 == ext2 else "txt")
    highlights = _get_highlights(text1, text2)

    w_score: float = float(winnowing_score)  # type: ignore[arg-type]
    s_score: float = float(struct_score)  # type: ignore[arg-type]
    e_score: float = float(sem_score)  # type: ignore[arg-type]
    t_score: float = float(total_score)  # type: ignore[arg-type]

    score_obj = ComparisonScore(
        winnowing_score=round(w_score, 4),
        jaccard_score=round(s_score, 4),
        semantic_score=round(e_score, 4),
        total_score=round(t_score, 4),
    )

    result = ComparisonResponse(
        scores=score_obj,
        highlights=highlights,
        message="Comparison completed successfully.",
        source_text=text1,
        target_text=text2,
    )

    return result


# ---------------------------------------------------------------------------
# Batch compare helpers
# ---------------------------------------------------------------------------
def _name_from_filename(filename: str) -> str:
    name, _ = os.path.splitext(_sanitize_filename(filename))
    return name.replace("_", " ").replace("-", " ").title()


def _validate_session_token(token: str) -> bool:
    secret = os.getenv("BATCH_WS_SECRET", "replace_with_random_32_char_string")
    return token == secret


def _build_summary(student_index: int, matrix: List[List[Optional[PairScore]]], n: int) -> StudentSummary:
    scores: List[float] = []
    for j in range(n):
        if student_index == j:
            continue
        cell = matrix[student_index][j]
        if cell is not None:
            scores.append(cell.total_score)
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

    return StudentSummary(student_index=student_index, max_score=max_score, avg_score=avg_score, risk_level=risk)


def _run_pair_sync(text_a, text_b, type_a, type_b):
    w = winnowing_similarity(text_a, text_b)
    s = structural_similarity(text_a, text_b, type_a if type_a == type_b else "txt")
    e = 0.0  # Semantic removed
    total = _aggregate(w, s, e, type_a if type_a == type_b else "txt")
    highlights = _get_highlights(text_a, text_b)

    w_f: float = float(w)  # type: ignore[arg-type]
    s_f: float = float(s)  # type: ignore[arg-type]
    e_f: float = float(e)  # type: ignore[arg-type]
    t_f: float = float(total)  # type: ignore[arg-type]

    return PairScore(
        winnowing_score=round(w_f, 4),
        jaccard_score=round(s_f, 4),
        semantic_score=round(e_f, 4),
        total_score=round(t_f, 4),
        highlights=highlights,
    )


# ---------------------------------------------------------------------------
# Batch compare (protected + rate limited)
# ---------------------------------------------------------------------------
@app.post("/compare-batch", response_model=BatchComparisonResponse)
@limiter.limit("5/minute")
async def compare_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
):
    n = len(files)
    if n < 2 or n > MAX_BATCH_FILES:
        raise HTTPException(400, detail=f"Requires 2–{MAX_BATCH_FILES} files")

    parsed = []
    texts = []

    for f in files:
        ext = _validate_file(f)
        content = await _read_validated(f)
        text = parse_file(f.filename, content)
        parsed.append((text, ext))
        texts.append(text)

    students = [
        StudentMeta(
            student_index=i,
            name=_name_from_filename(files[i].filename),
            filename=_sanitize_filename(files[i].filename),
        )
        for i in range(n)
    ]

    pairs = list(itertools.combinations(range(n), 2))
    total_pairs = len(pairs)
    matrix: List[List[Optional[PairScore]]] = [[None] * n for _ in range(n)]

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
            await queue.put({"completed": completed, "total": total_pairs, "current_pair": [i, j]})
        return i, j, res

    tasks = [
        process_pair(i, j, parsed[i][0], parsed[j][0], parsed[i][1], parsed[j][1])
        for i, j in pairs
    ]
    results = await asyncio.gather(*tasks)

    for i, j, res in results:
        matrix[i][j] = res
        matrix[j][i] = res

    summary = sorted(
        [_build_summary(i, matrix, n) for i in range(n)],
        key=lambda s: s.max_score,
        reverse=True,
    )

    if queue:
        await queue.put("DONE")

    batch_result = BatchComparisonResponse(students=students, matrix=matrix, summary=summary, parsed_texts=texts)

    return batch_result


# History endpoints removed out of scope


# ---------------------------------------------------------------------------
# WebSocket for batch progress
# ---------------------------------------------------------------------------
@app.websocket("/ws/batch-progress")
async def batch_progress_ws(ws: WebSocket, token: str = Query(...)):
    # HTTP CORS headers do NOT protect WebSocket handshakes — check Origin manually
    origin = ws.headers.get("origin", "")
    if "*" not in ALLOWED_ORIGINS and origin not in ALLOWED_ORIGINS:
        await ws.close(code=1008)
        return
    if not _validate_session_token(token):
        await ws.close(code=1008)
        return
    await ws.accept()

    session_id = None
    try:
        data = await ws.receive_json()
        session_id = data.get("session_id")

        if session_id:
            queue: asyncio.Queue = asyncio.Queue()
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
