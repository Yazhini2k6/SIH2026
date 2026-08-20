"""Unit tests for Stage 2 nlp_processing module."""

from sih2026.nlp_processing.page_classifier import classify_page
from sih2026.nlp_processing.pipeline import run_pipeline
from sih2026.nlp_processing.skill_extractor import extract_candidates


def test_page_classifier():
    admin_text = (
        "Course Code: CS101\n"
        "Credits: 4\n"
        "Internal Assessment: 40 Marks\n"
        "End Semester Exam: 60 Marks\n"
        "Attendance: 75% mandatory."
    )
    content_text = (
        "UNIT I: INTRODUCTION TO DATA SCIENCE\n"
        "Concepts of Big Data, Data Processing Pipeline, and Machine Learning Algorithms."
    )

    admin_decision = classify_page(1, admin_text)
    content_decision = classify_page(2, content_text)

    assert admin_decision.relevant is False
    assert content_decision.relevant is True


def test_skill_extractor():
    sentence = "Students will learn Python and Machine Learning concepts using TensorFlow."
    candidates = extract_candidates(1, sentence)

    assert len(candidates) > 0
    candidate_texts = [c.text for c in candidates]
    canonical_names = [c.canonical for c in candidates if c.canonical]

    assert "Python" in canonical_names or "Python" in candidate_texts
    assert "Machine Learning" in canonical_names or "Machine Learning" in candidate_texts


def test_run_pipeline_end_to_end():
    sample_stage1 = {
        "filename": "sample_syllabus.pdf",
        "total_pages": 2,
        "total_words": 50,
        "pages": [
            {
                "page_number": 1,
                "text": "Course Code: CS200\nCredits: 3\nInternal Assessment: 30 Marks",
                "extraction_method": "text",
                "word_count": 10,
                "lines": ["Course Code: CS200", "Credits: 3", "Internal Assessment: 30 Marks"],
            },
            {
                "page_number": 2,
                "text": "UNIT I: Python Programming and SQL Databases.\nIntroduction to Data Structures and Algorithms.",
                "extraction_method": "text",
                "word_count": 15,
                "lines": [
                    "UNIT I: Python Programming and SQL Databases.",
                    "Introduction to Data Structures and Algorithms.",
                ],
            },
        ],
        "llm_input_context": "--- DOCUMENT START: sample_syllabus.pdf ---",
    }

    result = run_pipeline(sample_stage1, verbose=False)

    assert result["filename"] == "sample_syllabus.pdf"
    assert "stats" in result
    assert result["stats"]["total_pages"] == 2
    assert result["stats"]["relevant_pages"] >= 1
    assert result["stats"]["final_candidates"] > 0

    skills = [c["text"] for c in result.get("skills", result.get("candidate_skills", []))]
    assert any("Python" in s for s in skills)
    assert any("SQL" in s for s in skills)
