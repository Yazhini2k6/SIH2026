"""Step 3 — Candidate skill extraction.

Runs three independent extraction methods over each kept sentence and
merges their output into one candidate list. Each method catches
things the others miss:

    - taxonomy match : exact / alias match against skill_taxonomy.json
                        (fast, zero false negatives for known terms)
    - GLiNER          : zero-shot entity extraction with custom labels
                        (catches skills not in the taxonomy)
    - YAKE            : unsupervised keyword/phrase extraction
                        (catches descriptive multi-word phrases like
                        "predictive modelling techniques")

Every candidate keeps its source method, source sentence and page
number for traceability — this is what lets the final system say
"Python appears on page 15" instead of just "Python: yes/no".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    from gliner import GLiNER
    _GLINER_AVAILABLE = True
except ImportError:
    _GLINER_AVAILABLE = False

try:
    import yake
    _YAKE_AVAILABLE = True
except ImportError:
    _YAKE_AVAILABLE = False


_TAXONOMY_PATH = Path(__file__).parent / "skill_taxonomy.json"

_GLINER_LABELS = [
    "programming language",
    "framework",
    "database",
    "ai concept",
    "devops tool",
    "cloud platform",
    "technical skill",
]

_gliner_cache = {"model": None}
_yake_cache = {"extractor": None}


@dataclass
class Candidate:
    text: str
    page_number: int
    source_sentence: str
    method: str          # "taxonomy" | "gliner" | "yake"
    canonical: str | None = None   # filled in if taxonomy match, else None
    score: float | None = None


def _load_taxonomy() -> dict:
    with open(_TAXONOMY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_taxonomy_lookup(taxonomy: dict) -> list[tuple[str, str]]:
    """Returns list of (surface_form_lowercase, canonical_name), including
    aliases, sorted longest-first so multi-word terms match before their
    substrings do (e.g. 'machine learning' before 'learning')."""
    pairs = []
    for canonical, meta in taxonomy.items():
        pairs.append((canonical.lower(), canonical))
        for alias in meta.get("aliases", []):
            pairs.append((alias.lower(), canonical))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


_TAXONOMY = _load_taxonomy()
_TAXONOMY_LOOKUP = _build_taxonomy_lookup(_TAXONOMY)


def extract_taxonomy_matches(page_number: int, sentence: str) -> list[Candidate]:
    lowered = sentence.lower()
    matches = []
    matched_spans: list[tuple[int, int]] = []

    for surface_form, canonical in _TAXONOMY_LOOKUP:
        # Word-boundary match to avoid "java" matching inside "javascript" etc.
        pattern = r"\b" + re.escape(surface_form) + r"\b"
        for m in re.finditer(pattern, lowered):
            span = (m.start(), m.end())
            # Skip if this span overlaps a longer match already found.
            if any(not (span[1] <= s or span[0] >= e) for s, e in matched_spans):
                continue
            matched_spans.append(span)
            matches.append(Candidate(
                text=sentence[m.start():m.end()],
                page_number=page_number,
                source_sentence=sentence,
                method="taxonomy",
                canonical=canonical,
                score=1.0,
            ))
    return matches


def _get_gliner():
    if not _GLINER_AVAILABLE:
        return None
    if _gliner_cache["model"] is None:
        # Small pretrained zero-shot NER model, runs locally, no training.
        _gliner_cache["model"] = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
    return _gliner_cache["model"]


def extract_gliner_matches(page_number: int, sentence: str, threshold: float = 0.4) -> list[Candidate]:
    model = _get_gliner()
    if model is None or not sentence or len(sentence.strip()) <= 3:
        return []
    try:
        entities = model.predict_entities(sentence[:500], _GLINER_LABELS, threshold=threshold)
        return [
            Candidate(
                text=e["text"],
                page_number=page_number,
                source_sentence=sentence,
                method="gliner",
                score=round(float(e.get("score", threshold)), 3),
            )
            for e in entities
        ]
    except Exception:
        return []


def extract_gliner_batch_matches(
    items: list[tuple[int, str]], threshold: float = 0.4, batch_size: int = 32
) -> list[Candidate]:
    """Runs GLiNER in batch mode across a list of (page_number, sentence) pairs."""
    model = _get_gliner()
    if model is None or not items:
        return []

    results: list[Candidate] = []
    # Truncate extremely long sentences to 500 chars to avoid memory/hang issues
    valid_items = [(pg, s[:500]) for pg, s in items if s and len(s.strip()) > 3]

    for i in range(0, len(valid_items), batch_size):
        chunk = valid_items[i : i + batch_size]
        sentences = [s for _, s in chunk]
        try:
            if hasattr(model, "batch_predict_entities"):
                batch_ents = model.batch_predict_entities(
                    sentences, _GLINER_LABELS, threshold=threshold
                )
            else:
                batch_ents = [
                    model.predict_entities(s, _GLINER_LABELS, threshold=threshold)
                    for s in sentences
                ]

            for (page_number, sentence), entities in zip(chunk, batch_ents):
                for e in entities:
                    results.append(
                        Candidate(
                            text=e["text"],
                            page_number=page_number,
                            source_sentence=sentence,
                            method="gliner",
                            score=round(float(e.get("score", threshold)), 3),
                        )
                    )
        except Exception:
            continue

    return results



def _get_yake():
    if not _YAKE_AVAILABLE:
        return None
    if _yake_cache["extractor"] is None:
        # ngram size 1-3 to catch multi-word technical phrases, top=3 per sentence.
        _yake_cache["extractor"] = yake.KeywordExtractor(
            lan="en", n=3, dedupLim=0.9, top=3, features=None
        )
    return _yake_cache["extractor"]


def extract_yake_matches(page_number: int, sentence: str) -> list[Candidate]:
    extractor = _get_yake()
    if extractor is None:
        return []
    keywords = extractor.extract_keywords(sentence)
    # YAKE scores are "lower is better" (0.0 = exact/salient, >0.15 = weak/noisy).
    # We filter raw scores <= 0.15 for high precision, then invert for score output.
    return [
        Candidate(
            text=kw,
            page_number=page_number,
            source_sentence=sentence,
            method="yake",
            score=round(1.0 / (1.0 + score), 3),
        )
        for kw, score in keywords
        if score <= 0.15
    ]



def extract_candidates(page_number: int, sentence: str) -> list[Candidate]:
    """Runs all three extraction methods on one sentence and returns the
    combined, unfiltered candidate list (deduplication happens in
    candidate_cleanup.py, not here — this step is deliberately
    high-recall / low-precision)."""
    candidates: list[Candidate] = []
    candidates.extend(extract_taxonomy_matches(page_number, sentence))
    candidates.extend(extract_gliner_matches(page_number, sentence))
    candidates.extend(extract_yake_matches(page_number, sentence))
    return candidates
