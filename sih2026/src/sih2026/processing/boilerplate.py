"""Boilerplate, header/footer, and page number removal module."""

import re


def is_page_number_line(line: str) -> bool:
    """Checks if a string is purely a page number indicator."""
    cleaned = line.strip().lower()
    if not cleaned:
        return False

    patterns = [
        r"^page\s+\d+(\s+of\s+\d+)?$",
        r"^\d+\s+of\s+\d+$",
        r"^\d+$",
        r"^-?\s*\d+\s*-?$",
        r"^\[?\s*page\s*\d+\s*\]?$",
    ]
    return any(re.match(pat, cleaned) for pat in patterns)


def remove_boilerplate(doc_json: dict, threshold_ratio: float = 0.5) -> dict:
    """Detects and removes repeating header/footer lines and page numbers across pages.

    Args:
        doc_json: Document payload matching Document schema (dict format).
        threshold_ratio: Minimum fraction of pages a line must appear on to be considered boilerplate.

    Returns:
        dict: Processed document payload with boilerplate removed.
    """
    pages = doc_json.get("pages", [])
    if not pages or len(pages) < 2:
        for page in pages:
            lines = page.get("text", "").split("\n")
            filtered = [line for line in lines if not is_page_number_line(line)]
            page["text"] = "\n".join(filtered)
            page["lines"] = [line for line in filtered if line.strip()]
        return doc_json

    total_pages = len(pages)
    top_candidates: dict[str, int] = {}
    bottom_candidates: dict[str, int] = {}

    for page in pages:
        p_lines = [l.strip() for l in page.get("text", "").split("\n") if l.strip()]
        if not p_lines:
            continue

        for top_line in p_lines[:2]:
            if not is_page_number_line(top_line):
                top_candidates[top_line] = top_candidates.get(top_line, 0) + 1

        for bottom_line in p_lines[-2:]:
            if not is_page_number_line(bottom_line):
                bottom_candidates[bottom_line] = bottom_candidates.get(bottom_line, 0) + 1

    boilerplate_lines = set()
    min_count = max(2, int(total_pages * threshold_ratio))

    for line_text, count in top_candidates.items():
        if count >= min_count:
            boilerplate_lines.add(line_text)

    for line_text, count in bottom_candidates.items():
        if count >= min_count:
            boilerplate_lines.add(line_text)

    for page in pages:
        p_lines = page.get("text", "").split("\n")
        cleaned_lines = []
        for line in p_lines:
            stripped = line.strip()
            if is_page_number_line(stripped):
                continue
            if stripped in boilerplate_lines:
                continue
            cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines).strip()
        page["text"] = cleaned_text
        page["lines"] = [l for l in cleaned_lines if l.strip()]
        page["word_count"] = len(cleaned_text.split())

    doc_json["total_words"] = sum(p.get("word_count", 0) for p in pages)

    filename = doc_json.get("filename", "document")
    blocks = [f"--- DOCUMENT START: {filename} ---"]
    for page in pages:
        blocks.append(f"\n[PAGE {page.get('page_number', 1)}]")
        blocks.append(page.get("text", ""))
    blocks.append("\n--- DOCUMENT END ---")
    doc_json["llm_input_context"] = "\n".join(blocks)

    return doc_json
