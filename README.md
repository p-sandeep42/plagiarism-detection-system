# 🔍 AuraDiff — Multi-Algorithm Plagiarism Detection System

AuraDiff is a full-stack plagiarism detection application that compares documents using a **three-engine analysis pipeline**: fingerprint-based detection, structural/AST comparison, and semantic similarity via transformer embeddings. It supports both one-to-one document comparison and batch analysis across an entire class submission set.

---

## ✨ Features

- **Pairwise Comparison** — Upload two files and get an instant, detailed similarity report with per-algorithm scores and highlighted matching segments
- **Batch Comparison** — Upload up to 30 files at once; the system runs every pairwise combination and surfaces a risk-ranked student summary matrix
- **Three-Engine Analysis** — Combines fingerprinting, structural (AST), and semantic similarity into a single weighted aggregate score
- **Smart Highlighting** — Matched passages are highlighted directly in a side-by-side split view, distinguishing exact vs. semantically-similar matches
- **Code-Aware** — Python and JavaScript files are compared at the AST level, not raw text, making it resistant to trivial renaming tricks
- **Multi-Format Parsing** — Accepts `.pdf`, `.docx`, `.txt`, `.py`, `.js`, `.md`, `.csv`, and `.json` files
- **Real-time Progress** — Batch jobs stream progress updates to the frontend over WebSocket

---

## 🏗️ Project Structure

```
final-plg/
├── backend/                        # Python FastAPI backend
│   ├── main.py                     # API entrypoint (FastAPI app, routes)
│   ├── algorithms/
│   │   ├── winnowing.py            # Fingerprint-based similarity (Winnowing algorithm)
│   │   ├── structural.py           # AST-based structural similarity (Python & JS)
│   │   └── semantic.py             # Semantic similarity via sentence-transformers
│   ├── services/
│   │   └── parser.py               # File parser (PDF, DOCX, TXT, source code)
│   ├── models/
│   │   └── schemas.py              # Pydantic response models
│   ├── requirements.txt            # Python dependencies
│   └── Dockerfile                  # Docker config for backend
│
├── frontend/                       # Next.js 14 frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx            # Landing / Dashboard page
│   │   │   ├── compare/            # Pairwise comparison page
│   │   │   ├── batch/              # Batch upload & results page
│   │   │   └── api/                # Next.js API proxy routes
│   │   ├── components/
│   │   │   ├── Dashboard.tsx       # Home dashboard component
│   │   │   ├── FileUpload.tsx      # Drag-and-drop file uploader
│   │   │   ├── SplitView.tsx       # Side-by-side text comparison view
│   │   │   └── batch/              # Batch result components (matrix, student panels)
│   │   ├── lib/                    # Utility functions & API client
│   │   └── types/                  # Shared TypeScript type definitions
│   ├── next.config.mjs
│   └── tailwind.config.ts
│
├── run.ps1                         # One-command launcher for both servers (Windows)
├── .gitignore
└── README.md
```

---

## 🧠 How the Analysis Works

### 1. 🔑 Winnowing Fingerprinting (Text Similarity)
Converts text into a set of k-gram hashes and selects minimum hashes within a sliding window. Similarity is measured as **Jaccard similarity** over the resulting fingerprint sets. This is highly effective at catching copy-paste plagiarism.

### 2. 🌲 Structural Similarity (AST Comparison)
For **Python** files, the code is parsed into an Abstract Syntax Tree (AST) and the sequence of AST node types is compared. For **JavaScript** files, `esprima` is used. For prose documents, Jaccard similarity over token sets is used instead. This approach catches structural plagiarism even when variable names or formatting have been changed.

### 3. 🧬 Semantic Similarity (Embedding Comparison)
Uses the `all-MiniLM-L6-v2` sentence-transformer model to generate dense vector embeddings for text chunks. Cosine similarity between embeddings is computed and max-pooled to produce an overall semantic score. This catches paraphrased content that evades keyword-level detection.

### 4. ⚖️ Weighted Aggregate Score
The three scores are combined with **file-type-aware weights**:

| File Type | Winnowing | Structural | Semantic |
|-----------|:---------:|:----------:|:--------:|
| Code (`.py`, `.js`, `.ts`, etc.) | 20% | 60% | 20% |
| Documents (`.txt`, `.pdf`, `.docx`, etc.) | 30% | 20% | 50% |

### 5. 🟡 Risk Classification (Batch Mode)
Each student in a batch is assigned a risk level based on their highest pairwise score:

| Score Range | Risk Level |
|-------------|-----------|
| < 30% | 🟢 Low |
| 30% – 60% | 🟡 Medium |
| 61% – 79% | 🟠 High |
| ≥ 80% | 🔴 Critical |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm

### 1. Clone the Repository

```bash
git clone https://github.com/p-sandeep42/plagiarism-detection-system.git
cd plagiarism-detection-system
```

### 2. Set Up the Backend

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Set Up the Frontend

```bash
cd frontend
npm install
```

### 4. Run the Application

#### Option A — One-Command Launch (Windows only)
From the project root:
```powershell
.\run.ps1
```
This starts both the backend (port `8000`) and frontend (port `3000`) simultaneously.

#### Option B — Manual Launch

**Backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Frontend (separate terminal):**
```bash
cd frontend
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/compare` | Compare two uploaded files |
| `POST` | `/compare-batch` | Batch compare up to 30 files |
| `WS` | `/ws/batch-progress` | WebSocket stream for batch job progress |

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server |
| **sentence-transformers** (`all-MiniLM-L6-v2`) | Semantic embeddings |
| **PyTorch** | Tensor operations for embedding similarity |
| **scikit-learn** | Cosine similarity matrix computation |
| **PyMuPDF** (`fitz`) | PDF text extraction |
| **python-docx** | DOCX text extraction |
| **esprima** | JavaScript AST parsing |
| **Pydantic** | Data validation & schemas |

### Frontend
| Technology | Purpose |
|------------|---------|
| **Next.js 14** | React framework (App Router) |
| **TypeScript** | Type-safe frontend code |
| **TailwindCSS** | Utility-first styling |
| **WebSocket** | Real-time batch progress streaming |

---

## 📄 Supported File Types

| Format | Extension(s) |
|--------|-------------|
| PDF documents | `.pdf` |
| Word documents | `.docx` |
| Plain text | `.txt` |
| Python source | `.py` |
| JavaScript source | `.js` |
| Markdown | `.md` |
| Data files | `.csv`, `.json` |

---

## 🔒 Privacy

All file processing happens **locally** on your machine. No documents are stored or transmitted to any external service. The semantic model (`all-MiniLM-L6-v2`) runs entirely offline after the initial one-time download from Hugging Face.

---

## 👤 Author

Developed by **Sandeep** — [github.com/p-sandeep42](https://github.com/p-sandeep42)
