export interface ComparisonScore {
  winnowing_score: number;
  jaccard_score: number;
  semantic_score: number;
  total_score: number;
}

export interface HighlightInfo {
  source_index_start: number;
  source_index_end: number;
  target_index_start: number;
  target_index_end: number;
  text: string;
  match_type: string;
}

export interface ComparisonResponse {
  scores: ComparisonScore;
  highlights: HighlightInfo[];
  message: string;
  source_text: string;
  target_text: string;
}

export interface StudentMeta {
  student_index: number;
  name: string;
  filename: string;
}

export interface PairScore {
  winnowing_score: number;
  jaccard_score: number;
  semantic_score: number;
  total_score: number;
  highlights: HighlightInfo[];
}

export interface StudentSummary {
  student_index: number;
  max_score: number;
  avg_score: number;
  risk_level: string;
}

export interface BatchComparisonResponse {
  students: StudentMeta[];
  matrix: (PairScore | null)[][];
  summary: StudentSummary[];
  parsed_texts: string[];
}
