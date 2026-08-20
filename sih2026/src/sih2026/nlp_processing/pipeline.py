"""Unified Production-Ready NLP Pipeline Orchestrator for SIH2026.

Executes Stage 2 NLP processing and Real Target-LLM Tokenization in exact logical order:
    1. Load Stage 1 JSON
    2. Page / section filtering
    3. Sentence splitting
    4. Skill extraction / NER (Taxonomy + GLiNER)
    5. Keyword extraction (YAKE)
    6. Skill classification (Taxonomy categorization)
    7. Deduplication / consolidation
    8. Generate ONE consolidated sylabus_nlp.json
    9. Tokenization using REAL target-LLM tokenizer (tiktoken cl100k_base / gpt-4o-mini) -> sylabus_nlp_tokenized.json

Usage:
    from sih2026.nlp_processing import run_pipeline
    nlp_result = run_pipeline(stage1_json)

    # CLI usage:
    python -m sih2026.nlp_processing.pipeline data/output/sylabus.json data/output
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .page_classifier import classify_pages
from .sentence_filter import filter_page_sentences
from .skill_extractor import (
    extract_taxonomy_matches,
    extract_yake_matches,
    extract_gliner_batch_matches,
    Candidate,
)
from .candidate_cleanup import clean_candidates
from .tokenizer import tokenize_nlp_output


def run_pipeline(
    stage1_json: dict,
    verbose: bool = False,
    tokenizer_model: str = "gpt-4o-mini",
    output_dir: Optional[Path] = None,
    save_files: bool = True,
    target_role: str = "Full-Stack Software Developer",
    provider: str = "auto",
    ollama_host: str = "",
    ollama_model: str = "",
    gemini_api_key: str = "",
    gemini_model: str = "gemini-2.5-flash",
) -> dict:
    """Runs the unified complete NLP pipeline, Real Target-LLM Tokenizer, and AI Skill-Gap Evaluator.

    Args:
        stage1_json: Dictionary containing Stage-1 Document schema ("pages", "filename", etc.)
        verbose: If True, prints step progress.
        tokenizer_model: Model name for real target-LLM tokenization (default: "gpt-4o-mini").
        output_dir: Directory where output files will be saved.
        save_files: If True, writes final JSON files to output_dir.
        target_role: Target industry role for gap comparison.
        provider: Provider for AI analysis ('auto', 'ollama', 'gemini', 'rule_based').
        ollama_host: Host URL for Ollama server.
        ollama_model: Ollama model identifier.
        gemini_api_key: Gemini API key.
        gemini_model: Gemini model identifier.

    Returns:
        Dictionary matching the consolidated final NLP schema including AI analysis.
    """
    filename = stage1_json.get("filename", "sylabus.pdf")
    stem = Path(filename).stem
    pages = stage1_json.get("pages", [])

    # --- Stage 2: Page classification & filtering ---
    page_decisions = classify_pages(pages)
    relevant_page_numbers = {d.page_number for d in page_decisions if d.relevant}
    relevant_pages = [p for p in pages if p.get("page_number") in relevant_page_numbers]

    if verbose:
        print(f"[1] Page classification: {len(relevant_pages)}/{len(pages)} pages kept")

    # --- Stage 3: Sentence splitting & filtering ---
    kept_sentences: list[tuple[int, str]] = []
    for page in relevant_pages:
        page_num = page.get("page_number", 0)
        decisions = filter_page_sentences(page_num, page.get("text", ""))
        for d in decisions:
            if d.keep:
                kept_sentences.append((d.page_number, d.sentence))

    if verbose:
        print(f"[2] Sentence filtering: {len(kept_sentences)} sentences kept")

    # --- Stage 4 & 5: Candidate Skill Extraction (Taxonomy + GLiNER) & Keyword Extraction (YAKE) ---
    raw_candidates: list[Candidate] = []
    keyword_set: set[str] = set()

    for page_number, sentence in kept_sentences:
        tax_matches = extract_taxonomy_matches(page_number, sentence)
        raw_candidates.extend(tax_matches)

        yake_matches = extract_yake_matches(page_number, sentence)
        raw_candidates.extend(yake_matches)
        for ym in yake_matches:
            if ym.text and len(ym.text) > 3:
                keyword_set.add(ym.text)

    gliner_matches = extract_gliner_batch_matches(kept_sentences, batch_size=32)
    raw_candidates.extend(gliner_matches)

    if verbose:
        print(f"[3] Skill & Keyword extraction: {len(raw_candidates)} raw candidates")

    # --- Stage 6 & 7: Skill Cleanup, Categorization, & Deduplication ---
    cleaned_candidates = clean_candidates(raw_candidates)

    # Group skills by taxonomy category for final classification schema
    classification_map: dict[str, list[str]] = {}
    skills_list: list[dict] = []

    for c in cleaned_candidates:
        skill_entry = {
            "text": c.text,
            "category": c.category,
            "pages": c.pages,
            "methods": c.methods,
            "occurrence_count": c.occurrence_count
        }
        skills_list.append(skill_entry)

        cat_key = c.category if c.category else "Uncategorized"
        classification_map.setdefault(cat_key, []).append(c.text)

    if verbose:
        print(f"[4] Cleanup & Consolidation: {len(cleaned_candidates)} deduplicated candidate skills")

    # --- Stage 8: Generate ONE Consolidated Final NLP JSON ---
    nlp_result = {
        "filename": filename,
        "document": {
            "name": filename,
            "source": stage1_json.get("source_type", "pdf"),
            "page_count": len(pages),
            "relevant_page_count": len(relevant_pages)
        },
        "stats": {
            "total_pages": len(pages),
            "relevant_pages": len(relevant_pages),
            "sentences_kept": len(kept_sentences),
            "raw_candidates": len(raw_candidates),
            "final_candidates": len(cleaned_candidates),
        },
        "relevant_pages": sorted(list(relevant_page_numbers)),
        "skills": skills_list,
        "keywords": sorted(list(keyword_set))[:100],  # Top 100 relevant keywords
        "classification": classification_map,
        "metadata": {
            "nlp_version": "1.0.0",
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_file": filename,
        }
    }

    # --- Stage 9: Tokenization using REAL Target-LLM Tokenizer ---
    token_source_name = f"{stem}_nlp.json"
    tokenized_result = tokenize_nlp_output(
        nlp_data=nlp_result,
        stage1_json=stage1_json,
        model_name=tokenizer_model,
        source_filename=token_source_name,
        include_token_ids=True
    )

    if verbose:
        metrics = tokenized_result["token_metrics"]
        print(
            f"[5] Tokenization ({tokenized_result['tokenizer']['name']}/{tokenizer_model}): "
            f"{metrics['original_token_count']} -> {metrics['nlp_token_count']} tokens "
            f"({metrics['reduction_percentage']}% reduction)"
        )

    # --- Stage 10: AI Skill-Gap Analysis ---
    from sih2026.evaluation.evaluator import SkillGapEvaluator
    evaluator = SkillGapEvaluator(
        provider=provider,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
    )
    gap_report = evaluator.evaluate(
        stage1_json=stage1_json,
        nlp_result=nlp_result,
        target_role=target_role
    )
    nlp_result["gap_analysis"] = gap_report

    if verbose:
        ai_info = gap_report.get("llm_analysis", {})
        score = ai_info.get("overall_alignment_score", 0)
        provider_name = gap_report.get("provider_used", "auto")
        print(f"[6] AI Skill-Gap Analysis ({provider_name} for '{target_role}'): Alignment Score = {score}/100")

    # Save outputs to disk
    if save_files:
        if output_dir is None:
            output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        nlp_file_path = output_dir / f"{stem}_nlp.json"
        with open(nlp_file_path, "w", encoding="utf-8") as f:
            json.dump(nlp_result, f, indent=2, ensure_ascii=False)

        tokenized_file_path = output_dir / f"{stem}_nlp_tokenized.json"
        with open(tokenized_file_path, "w", encoding="utf-8") as f:
            json.dump(tokenized_result, f, indent=2, ensure_ascii=False)

        gap_file_path = output_dir / f"{stem}_gap_analysis.json"
        with open(gap_file_path, "w", encoding="utf-8") as f:
            json.dump(gap_report, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"Wrote {nlp_file_path}")
            print(f"Wrote {tokenized_file_path}")
            print(f"Wrote {gap_file_path}")

    nlp_result["tokenization"] = tokenized_result
    return nlp_result


def _main():
    if len(sys.argv) < 2:
        print("Usage: python -m sih2026.nlp_processing.pipeline <stage1_output.json> [output_dir]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else in_path.parent

    with open(in_path, "r", encoding="utf-8") as f:
        stage1_json = json.load(f)

    print(f"Executing unified NLP pipeline on {in_path}...")
    nlp_res = run_pipeline(
        stage1_json,
        verbose=True,
        output_dir=out_dir,
        save_files=True
    )
    print("Done! Unified NLP pipeline complete.")


if __name__ == "__main__":
    _main()
