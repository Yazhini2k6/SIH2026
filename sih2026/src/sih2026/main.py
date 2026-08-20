"""Main entry point for document extraction, cleaning, NLP processing, tokenization, and AI Skill-Gap Analysis."""

import argparse
import json
import sys
from pathlib import Path

from sih2026.extraction.docx import extract_docx
from sih2026.extraction.exceptions import DocumentExtractionError
from sih2026.extraction.pdf import extract_pdf
from sih2026.nlp_processing import run_pipeline
from sih2026.processing.cleaner import clean_document


def process_document(
    input_path: Path,
    verbose: bool = False,
    target_role: str = "Full-Stack Software Developer",
    provider: str = "auto",
    ollama_host: str = "",
    ollama_model: str = "",
    gemini_api_key: str = "",
    gemini_model: str = "gemini-2.5-flash",
) -> Path:
    """Extracts, cleans, and converts a document into page-aware Stage-1 JSON,
    and runs Stage-2 NLP pipeline, tokenization, and Stage-3 AI Skill-Gap Analysis.

    Args:
        input_path: Path to input PDF or DOCX document.
        verbose: Print metrics if True.
        target_role: Industry target role for skill-gap comparison.
        provider: Provider identifier ('auto', 'ollama', 'gemini', 'rule_based').
        ollama_host: Host URL for Ollama server.
        ollama_model: Ollama model name.
        gemini_api_key: Gemini API Key.
        gemini_model: Gemini model identifier.

    Returns:
        Path to Stage-1 JSON output.
    """
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        raw_doc = extract_pdf(input_path)
    elif suffix == ".docx":
        raw_doc = extract_docx(input_path)
    else:
        raise ValueError(
            f"Unsupported file format '{input_path.suffix}'. Supported formats are .pdf and .docx"
        )

    cleaned_doc = clean_document(raw_doc)

    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_path.stem}.json"

    stage1_data = cleaned_doc.model_dump()
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stage1_data, f, indent=2, ensure_ascii=False)

    nlp_res = run_pipeline(
        stage1_json=stage1_data,
        verbose=verbose,
        output_dir=output_dir,
        save_files=True,
        target_role=target_role,
        provider=provider,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
    )

    return output_file


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("    python -m sih2026.main <document_path> [options]")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="SIH 2026 Skill-Gap Analyzer: Extraction, NLP, Tokenization & AI Evaluation"
    )
    parser.add_argument("document_path", type=str, help="Path to input PDF or DOCX syllabus document")
    parser.add_argument("--role", type=str, default="Full-Stack Software Developer", help="Target industry job role")
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "ollama", "gemini", "rule_based"], help="AI Provider")
    parser.add_argument("--host", type=str, default="", help="Ollama host URL (e.g. http://localhost:11434)")
    parser.add_argument("--model", type=str, default="", help="Ollama model identifier (e.g. llama3.1)")
    parser.add_argument("--api-key", type=str, default="", help="Gemini API Key")

    args = parser.parse_args()
    input_file = Path(args.document_path)

    try:
        output_file = process_document(
            input_file,
            verbose=True,
            target_role=args.role,
            provider=args.provider,
            ollama_host=args.host,
            ollama_model=args.model,
            gemini_api_key=args.api_key
        )
        stem = input_file.stem
        print(f"\nPipeline execution successful for '{input_file.name}':")
        print(f"  Stage-1 Document JSON:     data/output/{stem}.json")
        print(f"  Stage-2 Consolidated NLP:  data/output/{stem}_nlp.json")
        print(f"  Stage-2 LLM Tokenized:     data/output/{stem}_nlp_tokenized.json")
        print(f"  Stage-3 AI Skill-Gap JSON: data/output/{stem}_gap_analysis.json")
    except (FileNotFoundError, ValueError, DocumentExtractionError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
