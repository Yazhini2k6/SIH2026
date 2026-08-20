"""Step 4 — Candidate cleanup.

Takes the raw, noisy, high-recall candidate list from skill_extractor
and shrinks it to a compact, deduplicated list — this is the step that
actually delivers the token-reduction goal, since this compact list is
what gets sent to the LLM in Step 5 (not implemented in this package).

What happens here:
    1. Normalize text (strip, lowercase for comparison, keep original
       for display).
    2. POS filter — drop candidates that are just stray verbs/adjectives
       YAKE sometimes grabs (skills are almost always noun phrases).
       Only applied if spaCy is available; skipped otherwise.
    3. Merge taxonomy-matched candidates onto their canonical name
       immediately (no ambiguity there).
    4. Fuzzy-deduplicate the rest (e.g. "predictive modelling" vs
       "predictive modelling techniques") using simple string
       similarity — cheap, no embeddings needed at this stage.
    5. Collapse duplicates across pages into one entry with a merged
       `pages` list and all contributing source sentences kept (capped)
       for traceability.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from .skill_extractor import Candidate

try:
    import spacy
    _spacy_cleanup_cache = {"nlp": None}
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    _spacy_cleanup_cache = {"nlp": None}


def _get_spacy():
    if not _SPACY_AVAILABLE:
        return None
    if _spacy_cleanup_cache["nlp"] is None:
        try:
            # Enable NER for author/person name detection
            _spacy_cleanup_cache["nlp"] = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        except OSError:
            return None
    return _spacy_cleanup_cache["nlp"]


import re

_STOPWORD_LEAD_TRAIL = {
    "the", "a", "an", "and", "or", "of", "in", "on", "will", "using",
    "learn", "students", "course", "with", "for", "to",
}

_ADMIN_NOISE_TERMS = {
    "academic council", "course code", "internal assessment", "end semester",
    "end semester exam", "attendance", "regulation", "regulations", "prerequisite",
    "prerequisites", "marks", "credits", "evaluation pattern", "curriculum",
    "syllabus", "unit", "learning outcomes", "course objective", "course objectives",
    "semester", "hours", "lecture", "tutorial", "practical", "total marks",
    "passing marks", "examination", "exam pattern", "mit press", "wiley", "ieee",
    "springer", "mcgraw hill", "pearson", "edition", "vol.", "volume", "isbn",
    "ltd.", "inc.", "publishing company", "company ltd.", "pp.", "reading",
    "extensive reading", "revised edition", "prentice", "hall", "prentice hall",
    "publisher", "publishers", "author", "authors", "editor", "editors",
    "chapter", "section", "overview", "introduction", "complete reference",
    "reference manual", "pocket guide", "a practical approach", "case study"
}

_ACTION_VERBS = {
    "select", "apply", "develop", "choose", "determine", "create", "design",
    "implement", "construct", "calculate", "explain", "evaluate", "demonstrate",
    "compare", "analyze", "overview", "modifying", "printing", "covered",
    "decompose", "convert", "defining", "modifying", "understanding", "building",
    "request", "setting", "modifying"
}

_GENERIC_SINGLE_WORDS = {
    "data", "projects", "comm", "semantics", "overview", "collective", "para",
    "scope", "case", "study", "reference", "covered", "pub", "sons", "determine",
    "understand", "create", "modifying", "choose"
}

_PERSON_NAME_PATTERN = re.compile(
    r"^[A-Z][a-zA-Z'.\-]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-zA-Z'.\-]+$"
)

_FUZZY_DEDUPE_THRESHOLD = 0.82



def _has_duplicate_adjacent_words(text: str) -> bool:
    """Checks for adjacent duplicate words like 'LEXICAL ANALYZER Lexical' or 'INTRODUCTION Introduction'."""
    words = [w.lower().strip(".,;:()") for w in text.split()]
    for i in range(len(words) - 1):
        if words[i] and words[i] == words[i + 1]:
            return True
    return False


def _is_person_name(text: str, doc) -> bool:
    """Detects author / person names like 'Terrence W Pratt', 'Jean Paul', 'Cheng Liu'."""
    clean_t = text.strip()
    if _PERSON_NAME_PATTERN.match(clean_t):
        return True
    if doc is not None and len(doc.ents) > 0:
        for ent in doc.ents:
            if ent.label_ == "PERSON" and ent.text.strip() == clean_t:
                return True
    return False


def _is_admin_noise(text: str) -> bool:
    """Word/substring-level exclusion check for administrative/citation noise terms."""
    words = [w.strip(".,;:()") for w in text.lower().split()]
    padded_text = f" {' '.join(words)} "
    for term in _ADMIN_NOISE_TERMS:
        if f" {term} " in padded_text or term in words:
            return True
    return False


@dataclass
class CleanedSkill:
    text: str
    category: str | None
    pages: list[int] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    example_sentences: list[str] = field(default_factory=list)
    occurrence_count: int = 0


def _trim_stopword_edges(text: str) -> str:
    words = text.strip().split()
    while words and words[0].lower() in _STOPWORD_LEAD_TRAIL:
        words = words[1:]
    while words and words[-1].lower() in _STOPWORD_LEAD_TRAIL:
        words = words[:-1]
    return " ".join(words)


def _batch_pos_filter(texts: list[str]) -> set[str]:
    """Runs spaCy nlp.pipe across unique texts in high-throughput batches."""
    nlp = _get_spacy()
    unique_texts = list(set(texts))
    if nlp is None or not unique_texts:
        return set(unique_texts)

    valid_set = set()
    # Correct iteration: doc.text is built into spaCy's Doc object (prevents zip iterator misalignment!)
    for doc in nlp.pipe(unique_texts, batch_size=256):
        orig_text = doc.text
        if len(doc) == 0:
            continue

        # Drop adjacent duplicate word fragments (e.g. "LEXICAL ANALYZER Lexical", "INTRODUCTION Introduction")
        if _has_duplicate_adjacent_words(orig_text):
            continue

        # Drop person / author names (e.g. "Terrence W Pratt", "Jean Paul", "Sangeeta Sharma", "Michael McCarthy")
        if _is_person_name(orig_text, doc):
            continue

        # Drop phrases starting with action verbs (e.g. "Select and apply", "Apply sorting", "Develop", "Apply the concepts")
        first_word_lower = doc[0].text.lower()
        if first_word_lower in _ACTION_VERBS or doc[0].pos_ in ("VERB", "AUX"):
            continue

        has_noun = any(tok.pos_ in ("NOUN", "PROPN") for tok in doc)
        if not has_noun:
            continue

        last_pos = doc[-1].pos_
        if last_pos in ("VERB", "CCONJ", "SCONJ", "ADP", "AUX", "PUNCT"):
            continue

        valid_set.add(orig_text)

    return valid_set



def _stem_key(text: str) -> str:
    """Normalize plural endings and whitespace for fuzzy/exact comparison."""
    words = text.lower().strip().split()
    stemmed = []
    for w in words:
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        stemmed.append(w)
    return " ".join(stemmed)


def _is_fuzzy_duplicate(a: str, b: str) -> bool:
    a_stem = _stem_key(a)
    b_stem = _stem_key(b)
    
    if a_stem == b_stem:
        return True
        
    # Check letter sequence similarity
    ratio = difflib.SequenceMatcher(None, a_stem, b_stem).ratio()
    if ratio >= _FUZZY_DEDUPE_THRESHOLD:
        return True

    # Check token set equality (order-insensitive)
    a_tokens = set(a_stem.split())
    b_tokens = set(b_stem.split())
    if a_tokens == b_tokens:
        return True

    return False


def clean_candidates(candidates: list[Candidate], max_example_sentences: int = 3) -> list[CleanedSkill]:
    # Step A: normalize + drop empties + substring noise check + batch POS filter
    pre_cleaned: list[Candidate] = []
    texts_to_check: list[str] = []
    for c in candidates:
        clean_text = _trim_stopword_edges(c.text).strip()
        if not clean_text or len(clean_text) < 2:
            continue
        if _is_admin_noise(clean_text):
            continue
        c.text = clean_text
        pre_cleaned.append(c)
        if c.method != "taxonomy":
            texts_to_check.append(clean_text)

    valid_pos_texts = _batch_pos_filter(texts_to_check)

    normalized: list[Candidate] = []
    for c in pre_cleaned:
        if c.method == "taxonomy" or c.text in valid_pos_texts:
            normalized.append(c)

    # Step B: merge candidates onto canonical taxonomy terms & deduplicate
    merged: dict[str, CleanedSkill] = {}
    stem_to_key: dict[str, str] = {}
    token_set_to_key: dict[frozenset, str] = {}
    prefix_to_key: dict[str, str] = {}
    from .skill_extractor import _TAXONOMY

    def _key_for(c: Candidate) -> str | None:
        if c.canonical:
            return c.canonical.lower()

        c_stem = _stem_key(c.text)
        if c_stem in stem_to_key:
            return stem_to_key[c_stem]

        tokens = frozenset(c_stem.split())
        if tokens in token_set_to_key:
            return token_set_to_key[tokens]

        if len(c_stem.split()) >= 2:
            prefix = " ".join(c_stem.split()[:2])
            if prefix in prefix_to_key:
                return prefix_to_key[prefix]

        return None

    for c in normalized:
        key = _key_for(c)
        if key is None:
            key = c.canonical.lower() if c.canonical else _stem_key(c.text)

        c_stem = _stem_key(c.text)
        stem_to_key[c_stem] = key
        
        tokens = frozenset(c_stem.split())
        token_set_to_key[tokens] = key
        
        if len(c_stem.split()) >= 2:
            prefix = " ".join(c_stem.split()[:2])
            if prefix not in prefix_to_key:
                prefix_to_key[prefix] = key

        if key not in merged:
            category = None
            display_name = c.canonical or c.text
            if c.canonical and c.canonical in _TAXONOMY:
                category = _TAXONOMY[c.canonical]["category"]
            merged[key] = CleanedSkill(text=display_name, category=category)

        entry = merged[key]
        if c.page_number not in entry.pages:
            entry.pages.append(c.page_number)
        if c.method not in entry.methods:
            entry.methods.append(c.method)
        if len(entry.example_sentences) < max_example_sentences and c.source_sentence not in entry.example_sentences:
            entry.example_sentences.append(c.source_sentence)
        entry.occurrence_count += 1

    # Step C: Filter out single-occurrence and generic single-word YAKE-only keywords
    results: list[CleanedSkill] = []
    for r in merged.values():
        r.pages.sort()
        is_yake_only = (r.methods == ["yake"])
        words = r.text.strip().split()
        
        # Rule 1: Drop generic single-word terms without taxonomy category or GLiNER validation
        if len(words) == 1 and is_yake_only and r.category is None:
            continue

        # Rule 2: Drop generic single-word list matches
        if r.text.lower() in _GENERIC_SINGLE_WORDS and r.category is None:
            continue

        # Rule 3: Unsupervised multi-word YAKE keywords without taxonomy/GLiNER agreement must appear >= 2 times
        if is_yake_only and r.occurrence_count < 2:
            continue

        results.append(r)

    # Surface strongest-signal candidates first
    results.sort(key=lambda r: (len(r.methods), r.occurrence_count), reverse=True)
    return results



