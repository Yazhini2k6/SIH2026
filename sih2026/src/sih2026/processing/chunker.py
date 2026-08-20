"""Unit and module-aware text chunking module."""

import re


def detect_unit_headers(lines: list[str]) -> list[tuple[int, str]]:
    """Identifies line indices and titles of unit/module/chapter headers."""
    unit_pattern = r"^(UNIT|MODULE|CHAPTER|PART|SECTION)\s+([I|V|X|L|C|D|M]+|\d+)"
    headers = []

    for idx, line in enumerate(lines):
        line_clean = line.strip()
        if re.match(unit_pattern, line_clean, re.IGNORECASE):
            headers.append((idx, line_clean))

    return headers


def chunk_unit_aware(
    doc_json: dict,
    max_words_per_chunk: int = 400,
    overlap_words: int = 50,
) -> list[dict]:
    """Splits document text into unit-aware chunks with sliding window overlap.

    Args:
        doc_json: Document payload dictionary matching Document schema.
        max_words_per_chunk: Maximum target word count per chunk.
        overlap_words: Number of overlapping words to carry forward between chunks.

    Returns:
        list[dict]: List of chunk objects with unit titles, page ranges, and text.
    """
    pages = doc_json.get("pages", [])
    if not pages:
        return []

    all_lines: list[tuple[int, str]] = []
    for page in pages:
        p_num = page.get("page_number", 1)
        p_text = page.get("text", "")
        for line in p_text.split("\n"):
            if line.strip():
                all_lines.append((p_num, line.strip()))

    if not all_lines:
        return []

    just_lines = [item[1] for item in all_lines]
    unit_headers = detect_unit_headers(just_lines)

    chunks: list[dict] = []
    chunk_id = 1

    if unit_headers:
        for i in range(len(unit_headers)):
            start_idx, unit_title = unit_headers[i]
            end_idx = (
                unit_headers[i + 1][0] if i + 1 < len(unit_headers) else len(all_lines)
            )

            unit_lines = all_lines[start_idx:end_idx]
            unit_pages = sorted(list({item[0] for item in unit_lines}))
            unit_text_lines = [item[1] for item in unit_lines]
            unit_text = "\n".join(unit_text_lines)
            words = unit_text.split()

            if len(words) > max_words_per_chunk:
                sub_start = 0
                sub_id = 1
                while sub_start < len(words):
                    sub_end = min(sub_start + max_words_per_chunk, len(words))
                    sub_words = words[sub_start:sub_end]
                    sub_text = " ".join(sub_words)

                    chunks.append(
                        {
                            "chunk_id": f"chunk_{chunk_id}_{sub_id}",
                            "unit_title": f"{unit_title} (Part {sub_id})",
                            "pages": unit_pages,
                            "text": sub_text,
                            "word_count": len(sub_words),
                        }
                    )
                    sub_id += 1
                    sub_start += max_words_per_chunk - overlap_words
            else:
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_id}",
                        "unit_title": unit_title,
                        "pages": unit_pages,
                        "text": unit_text,
                        "word_count": len(words),
                    }
                )
            chunk_id += 1
    else:
        sub_start = 0
        all_words = " ".join(just_lines).split()
        all_pages = sorted(list({item[0] for item in all_lines}))

        while sub_start < len(all_words):
            sub_end = min(sub_start + max_words_per_chunk, len(all_words))
            sub_words = all_words[sub_start:sub_end]
            sub_text = " ".join(sub_words)

            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_id}",
                    "unit_title": "General Content",
                    "pages": all_pages,
                    "text": sub_text,
                    "word_count": len(sub_words),
                }
            )
            chunk_id += 1
            sub_start += max_words_per_chunk - overlap_words

    return chunks
