from pydantic import BaseModel
from typing import List, Dict, Optional

class ComparisonScore(BaseModel):
    winnowing_score: float
    jaccard_score: float
    semantic_score: float
    total_score: float

class HighlightInfo(BaseModel):
    source_index_start: int
    source_index_end: int
    target_index_start: int
    target_index_end: int
    text: str
    match_type: str # "exact", "structural", "semantic"

class ComparisonResponse(BaseModel):
    scores: ComparisonScore
    highlights: List[HighlightInfo]
    message: str
    source_text: str = ""
    target_text: str = ""

class StudentMeta(BaseModel):
    student_index: int
    name: str
    filename: str

class PairScore(BaseModel):
    winnowing_score: float
    jaccard_score: float
    semantic_score: float
    total_score: float
    highlights: List[HighlightInfo]

class StudentSummary(BaseModel):
    student_index: int
    max_score: float
    avg_score: float
    risk_level: str

class BatchComparisonResponse(BaseModel):
    students: List[StudentMeta]
    matrix: List[List[Optional[PairScore]]]
    summary: List[StudentSummary]
    parsed_texts: List[str]
