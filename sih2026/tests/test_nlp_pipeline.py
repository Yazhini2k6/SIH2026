"""Unit tests for the unified NLP processing orchestrator and real LLM tokenizer."""

import json
from pathlib import Path
import pytest

from sih2026.nlp_processing import run_pipeline, count_and_tokenize, tokenize_nlp_output


@pytest.fixture
def sample_stage1_json():
    return {
        "filename": "sample_course.pdf",
        "source_type": "pdf",
        "llm_input_context": "--- DOCUMENT START: sample_course.pdf ---\n\n[PAGE 1]\nUNIT I: MACHINE LEARNING AND DATA SCIENCE\nStudents will learn Python, Machine Learning, and SQL.\n\n[PAGE 2]\nUNIT II: DEVOPS AND CLOUD COMPUTING\nTopics include Docker, Kubernetes, and AWS.",
        "pages": [
            {
                "page_number": 1,
                "text": "UNIT I: MACHINE LEARNING AND DATA SCIENCE\nStudents will learn Python, Machine Learning, and SQL.",
                "word_count": 14,
                "extraction_method": "text"
            },
            {
                "page_number": 2,
                "text": "UNIT II: DEVOPS AND CLOUD COMPUTING\nTopics include Docker, Kubernetes, and AWS.",
                "word_count": 12,
                "extraction_method": "text"
            }
        ]
    }


def test_real_llm_tokenizer_tiktoken():
    text = "Students will learn Python, Machine Learning, and SQL."
    count, token_ids, meta = count_and_tokenize(text, model_name="gpt-4o-mini", include_token_ids=True)

    assert count > 0
    assert meta["name"] in ("tiktoken", "transformers", "heuristic_fallback")
    assert meta["model"] == "gpt-4o-mini"
    assert isinstance(token_ids, list)
    assert len(token_ids) == count


def test_unified_nlp_pipeline_end_to_end(sample_stage1_json, tmp_path):
    nlp_res = run_pipeline(
        stage1_json=sample_stage1_json,
        verbose=False,
        output_dir=tmp_path,
        save_files=True
    )

    # Verify nlp_result structure
    assert nlp_res["filename"] == "sample_course.pdf"
    assert nlp_res["document"]["name"] == "sample_course.pdf"
    assert nlp_res["document"]["page_count"] == 2
    assert "skills" in nlp_res
    assert "keywords" in nlp_res
    assert "classification" in nlp_res
    assert "stats" in nlp_res
    assert "tokenization" in nlp_res

    # Verify output JSON files created on disk
    nlp_file = tmp_path / "sample_course_nlp.json"
    tok_file = tmp_path / "sample_course_nlp_tokenized.json"

    assert nlp_file.exists()
    assert tok_file.exists()

    # Verify tokenized result metrics
    tok_res = nlp_res["tokenization"]
    assert tok_res["source_file"] == "sample_course_nlp.json"
    assert "token_metrics" in tok_res
    metrics = tok_res["token_metrics"]
    assert metrics["original_token_count"] > 0
    assert metrics["nlp_token_count"] > 0
    assert metrics["reduction_tokens"] >= 0
    assert 0.0 <= metrics["reduction_percentage"] <= 100.0
