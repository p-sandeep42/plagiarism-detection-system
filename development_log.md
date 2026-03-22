# AuraDiff Development Log & Context

This file serves as a record of changes, findings, and current context for the AuraDiff (Plagiarism Detection System) project.

## Project Overview
AuraDiff is a multi-algorithmic plagiarism detection system featuring:
- **Fingerprinting (Winnowing)**: For exact match detection.
- **Structural Similarity (Jaccard)**: Using AST/token-based analysis for code.
- **Semantic Analysis**: Using machine learning models (SBERT) for paraphrasing detection.

## Project Structure
- `backend/`: FastAPI-based REST API handling the comparison logic.
  - `main.py`: Main API entry point.
  - `algorithms/`: Implementation of the various similarity algorithms.
  - `services/`: Helper services (e.g., file parsing).
- `frontend/`: Next.js-based web interface.
  - `src/app/`: Next.js 14 (App Router) structure.
  - `src/components/`: UI components (FileUpload, Comparison View, etc.).
  - `src/lib/backends/`: Integration with the FastAPI backend.

## Research Findings (2026-03-20)
### API Usage
- **FastAPI**: The project defines its own internal API at `http://localhost:8000`.
- **Next.js API Routes**: Used as a proxy/gateway (e.g., `/api/compare-batch`).
- **Semantic Analysis**: Performed locally using the `sentence-transformers` library (model: `all-MiniLM-L6-v2`). No external cloud APIs (like OpenAI) are currently used for processing.

## Recent Changes
- [x] Initial research on API usage and project architecture.
- [x] Identification of core components and communication flow.

## Current Context
- **Workspace**: `c:\Users\HP\OneDrive\Desktop\final-plg`
- **Active Backend**: FastAPI (Python)
- **Active Frontend**: Next.js (TypeScript)
- **Next Steps**: Continue development/debugging based on user requests.
