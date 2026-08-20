"""Step 2 — Sentence filter.

Within a page that Step 1 already marked as relevant, individual
sentences can still be noise (a stray "Course Code: CS301" embedded in
an otherwise content-heavy page). This step keeps only the sentences
worth passing on to skill extraction.

Same two-layer design as page_classifier: cheap regex rules first,
embedding similarity for anything ambiguous, keyword fallback if
sentence-transformers isn't installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .page_classifier import _ADMIN_RE, _get_model, _REFERENCE_LABELS, _EMBEDDINGS_AVAILABLE

try:
    from sentence_transformers import util as st_util
except ImportError:
    st_util = None

try:
    import spacy
    _nlp_cache = {"nlp": None}
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    _nlp_cache = {"nlp": None}


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _get_spacy():
    if not _SPACY_AVAILABLE:
        return None
    if _nlp_cache["nlp"] is None:
        try:
            _nlp_cache["nlp"] = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except OSError:
            # Model not downloaded (`python -m spacy download en_core_web_sm`)
            return None
    return _nlp_cache["nlp"]


def split_sentences(text: str) -> list[str]:
    """Sentence splitting. Uses spaCy if available for better accuracy on
    academic text with abbreviations (e.g. 'e.g.', 'i.e.', 'Fig.'),
    otherwise falls back to a regex splitter."""
    if not text or not text.strip():
        return []

    nlp = _get_spacy()
    if nlp is not None:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]

    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


@dataclass
class SentenceDecision:
    page_number: int
    sentence: str
    keep: bool
    method: str


def _is_obviously_admin(sentence: str) -> bool:
    return bool(_ADMIN_RE.search(sentence))


def _too_short_or_empty(sentence: str) -> bool:
    return len(sentence.split()) < 3


def filter_sentence(page_number: int, sentence: str) -> SentenceDecision:
    if _too_short_or_empty(sentence):
        return SentenceDecision(page_number, sentence, False, "rule")
    if _is_obviously_admin(sentence):
        return SentenceDecision(page_number, sentence, False, "rule")

    model = _get_model()
    if model is None or st_util is None:
        # Fallback: keep anything that passed the rule checks.
        return SentenceDecision(page_number, sentence, True, "keyword_fallback")

    sent_emb = model.encode(sentence, convert_to_tensor=True)
    admin_emb = model.encode(_REFERENCE_LABELS["administrative"], convert_to_tensor=True)
    content_emb = model.encode(_REFERENCE_LABELS["content"], convert_to_tensor=True)

    admin_sim = float(st_util.cos_sim(sent_emb, admin_emb))
    content_sim = float(st_util.cos_sim(sent_emb, content_emb))

    return SentenceDecision(page_number, sentence, content_sim > admin_sim, "embedding")


def filter_page_sentences(page_number: int, text: str) -> list[SentenceDecision]:
    sentences = split_sentences(text)
    return [filter_sentence(page_number, s) for s in sentences]
