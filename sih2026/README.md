# SIH 2026 - Skill-Gap Analyzer: Stage 1 Document Extraction & Cleaning

An AI-based Skill-Gap Analyzer designed to compare academic syllabi with industry requirements. 

This repository contains **Stage 1** of the pipeline: reliable, page-aware text extraction, conservative text cleaning, and **Local LLM-optimized JSON formatting** for **PDF** and **DOCX** documents.

---

## 📌 Architectural Pipeline

```
PDF / DOCX Document
        ↓
Text Extraction (PyMuPDF / python-docx)
        ↓
Page-Aware Document Representation (Pydantic Models)
        ↓
Conservative Text Cleaning (Null char removal, space normalization)
        ↓
Local LLM Prompt Formatting (Metadata, Page Lines & Delimited Context)
        ↓
LLM-Ready JSON Output
```

---

## 📂 Project Structure

```
sih2026/
├── data/
│   ├── input/             # Place input PDF and DOCX documents here
│   └── output/            # Generated LLM-formatted JSON output files
├── src/
│   └── sih2026/
│       ├── __init__.py
│       ├── main.py        # CLI entry point for the pipeline
│       ├── extraction/    # Extractor modules for PDF & DOCX
│       │   ├── __init__.py
│       │   ├── docx.py
│       │   ├── exceptions.py
│       │   └── pdf.py
│       ├── models/        # Pydantic data models
│       │   ├── __init__.py
│       │   └── document.py
│       └── processing/    # Conservative text cleaner & LLM formatter
│           ├── __init__.py
│           └── cleaner.py
├── tests/                 # Comprehensive Pytest suite
│   ├── test_cleaner.py
│   ├── test_document.py
│   ├── test_docx.py
│   ├── test_main.py
│   └── test_pdf.py
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependency tree
└── README.md
```

---

## 🚀 Quick Start & How to Run

### Prerequisites
- [uv](https://github.com/astral-sh/uv) package manager installed.
- Python `>= 3.14` (or compatible Python version managed by `uv`).

### 1. Install Dependencies
Dependencies are managed automatically with `uv`. To sync the environment:
```bash
uv sync
```

### 2. Run Document Processing

Run the main script against any `.pdf` or `.docx` file in `data/input/`:

```bash
# Process a PDF file
uv run python -m sih2026.main data/input/syllabus.pdf

# Process a DOCX file
uv run python -m sih2026.main data/input/syllabus.docx
```

Or via CLI entry point:
```bash
uv run sih2026 data/input/syllabus.pdf
```

---

## 🤖 Local LLM-Optimized JSON Schema

The generated JSON file under `data/output/<filename_stem>.json` is specifically structured to feed directly into local LLM models (e.g., Ollama, Llama 3, Qwen, DeepSeek, Mistral) for skill-gap inference.

**Example `data/output/syllabus.json`:**
```json
{
  "filename": "syllabus.pdf",
  "total_pages": 2,
  "total_words": 30,
  "pages": [
    {
      "page_number": 1,
      "text": "UNIT I: INTRODUCTION TO DATA SCIENCE\nConcepts of Big Data, Data Processing Pipeline.\nTools and Frameworks.",
      "extraction_method": "text",
      "word_count": 16,
      "lines": [
        "UNIT I: INTRODUCTION TO DATA SCIENCE",
        "Concepts of Big Data, Data Processing Pipeline.",
        "Tools and Frameworks."
      ]
    },
    {
      "page_number": 2,
      "text": "UNIT II: MACHINE LEARNING ALGORITHMS\nSupervised vs Unsupervised Learning.\nDecision Trees & Random Forests.",
      "extraction_method": "text",
      "word_count": 14,
      "lines": [
        "UNIT II: MACHINE LEARNING ALGORITHMS",
        "Supervised vs Unsupervised Learning.",
        "Decision Trees & Random Forests."
      ]
    }
  ],
  "llm_input_context": "--- DOCUMENT START: syllabus.pdf ---\n\n[PAGE 1]\nUNIT I: INTRODUCTION TO DATA SCIENCE\nConcepts of Big Data, Data Processing Pipeline.\nTools and Frameworks.\n\n[PAGE 2]\nUNIT II: MACHINE LEARNING ALGORITHMS\nSupervised vs Unsupervised Learning.\nDecision Trees & Random Forests.\n\n--- DOCUMENT END ---"
}
```

### 💡 Using the JSON Output with Local LLMs (e.g. Ollama / Python)

```python
import json
import requests

# 1. Load the generated JSON
with open("data/output/syllabus.json", "r", encoding="utf-8") as f:
    doc_data = json.load(f)

# 2. Extract the ready-to-use LLM input context
prompt_context = doc_data["llm_input_context"]

# 3. Formulate the prompt for your local LLM
system_prompt = "You are an expert curriculum evaluator. Analyze the syllabus context and identify key skills and potential industry skill gaps."
user_prompt = f"{prompt_context}\n\nTask: List all technical skills taught in this curriculum, organized by page number."

# 4. Call Local Ollama endpoint
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "stream": False
    }
)

print(response.json()["response"])
```

---

## 🛠️ How to Add New File Extensions & Extractors

To extend support for additional file types (e.g., `.txt`, `.pptx`, `.html`):

### Step 1: Create Extractor (`src/sih2026/extraction/txt.py`)
```python
from pathlib import Path
from sih2026.models.document import Document, DocumentPage
from sih2026.extraction.exceptions import UnsupportedFileTypeError, NoTextExtractedError

def extract_txt(file_path: str | Path) -> Document:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".txt":
        raise UnsupportedFileTypeError(f"Expected .txt file, got '{path.suffix}'")
        
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise NoTextExtractedError(f"TXT file '{path}' is empty.")
        
    return Document(
        filename=path.name,
        pages=[DocumentPage(page_number=1, text=text, extraction_method="txt")]
    )
```

### Step 2: Export Extractor (`src/sih2026/extraction/__init__.py`)
```python
from sih2026.extraction.txt import extract_txt
```

### Step 3: Register Handler (`src/sih2026/main.py`)
```python
elif suffix == ".txt":
    raw_doc = extract_txt(input_path)
```

---

## 🧪 Running Tests

Run the full pytest suite:
```bash
uv run pytest
```
