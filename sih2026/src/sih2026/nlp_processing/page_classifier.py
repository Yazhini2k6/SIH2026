"""Step 1 — Page classifier.

Decides whether a whole page from the Stage-1 JSON is worth processing
further (course content / syllabus / learning outcomes) or should be
discarded (regulations, credits, examination pattern, attendance rules).

Design:
    - A fast, cheap regex rule pass runs first and can short-circuit an
      obviously-administrative page without ever touching a model.
    - Anything the rules don't confidently reject is scored against a
      small set of reference "prototype" sentences using sentence
      embeddings (sentence-transformers). Whichever prototype the page
      is closest to decides relevant vs irrelevant.
    - If sentence-transformers isn't installed, we fall back to a
      keyword-density rule so the pipeline still runs end to end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False


# --- Rule layer -------------------------------------------------------

_ADMIN_PATTERNS = [
    r"\bcredits?\s*:?\s*\d",
    r"\binternal\s+assessment\b",
    r"\bend\s+semester\s+exam",
    r"\battendance\b",
    r"\bcourse\s+code\s*:?\s*[a-z]{2,4}\s*\d{2,4}",
    r"\bmarks?\s*:?\s*\d+",
    r"\bregulation[s]?\s*\d{2,4}\b",
    r"\bprerequisite[s]?\s*:",
]
_ADMIN_RE = re.compile("|".join(_ADMIN_PATTERNS), re.IGNORECASE)

_CONTENT_HINT_PATTERNS = [
    r"\bunit\s+[ivx\d]+\b",
    r"\bstudents?\s+will\b",
    r"\bconcepts?\s+of\b",
    r"\bintroduction\s+to\b",
    r"\btopics?\s+covered\b",
]
_CONTENT_HINT_RE = re.compile("|".join(_CONTENT_HINT_PATTERNS), re.IGNORECASE)


# --- Embedding layer ----------------------------------------------------

_REFERENCE_LABELS = {
    "administrative": (
        "Course credits, internal assessment marks, examination pattern, "
        "attendance rules and administrative regulations."
    ),
    "content": (
        "Course syllabus units, topics covered, learning outcomes and "
        "technical subject matter taught to students."
    ),
}

_model_cache: dict = {"model": None}


def _get_model():
    if not _EMBEDDINGS_AVAILABLE:
        return None
    if _model_cache["model"] is None:
        # Small, fast, local model — no API calls, ~80MB download once.
        _model_cache["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache["model"]


@dataclass
class PageClassification:
    page_number: int
    relevant: bool
    method: str          # "rule" or "embedding" or "keyword_fallback"
    score: Optional[float] = None


def _keyword_density_fallback(text: str) -> bool:
    """Used only if sentence-transformers isn't installed."""
    admin_hits = len(_ADMIN_RE.findall(text))
    content_hits = len(_CONTENT_HINT_RE.findall(text))
    word_count = max(len(text.split()), 1)
    # Longer pages with technical-sounding density are likely content pages.
    return content_hits >= admin_hits and word_count > 15


def classify_page(page_number: int, text: str) -> PageClassification:
    if not text or not text.strip():
        return PageClassification(page_number, False, "rule", 0.0)

    # Rule short-circuit: heavy admin pattern density and no content hints.
    admin_hits = len(_ADMIN_RE.findall(text))
    content_hits = len(_CONTENT_HINT_RE.findall(text))
    if admin_hits >= 3 and content_hits == 0:
        return PageClassification(page_number, False, "rule", 0.0)
    if content_hits >= 2 and admin_hits == 0:
        return PageClassification(page_number, True, "rule", 1.0)

    model = _get_model()
    if model is None:
        keep = _keyword_density_fallback(text)
        return PageClassification(page_number, keep, "keyword_fallback")

    page_emb = model.encode(text, convert_to_tensor=True)
    admin_emb = model.encode(_REFERENCE_LABELS["administrative"], convert_to_tensor=True)
    content_emb = model.encode(_REFERENCE_LABELS["content"], convert_to_tensor=True)

    admin_sim = float(st_util.cos_sim(page_emb, admin_emb))
    content_sim = float(st_util.cos_sim(page_emb, content_emb))

    relevant = content_sim > admin_sim
    return PageClassification(
        page_number, relevant, "embedding", round(content_sim - admin_sim, 4)
    )


def classify_pages(pages: list[dict]) -> list[PageClassification]:
    """pages: the 'pages' list from the Stage-1 JSON, each with
    'page_number' and 'text'."""
    return [classify_page(p["page_number"], p.get("text", "")) for p in pages]
