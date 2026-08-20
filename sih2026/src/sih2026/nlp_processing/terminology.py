"""Dictionary-based Terminology Lookup and Acronym Expansion Module."""

import json
import re
from pathlib import Path

ACRONYMS_FILE = Path(__file__).parent / "acronyms.json"


def load_acronyms() -> dict[str, str]:
    """Loads acronym dictionary dataset from local JSON storage."""
    if not ACRONYMS_FILE.exists():
        return {}
    with open(ACRONYMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_acronyms(input_data: list[str] | str) -> dict[str, str]:
    """Resolves technical abbreviations into expanded terminology names.

    Args:
        input_data: String text or list of acronym tokens (e.g. ['DBMS', 'K8s', 'DL']).

    Returns:
        dict[str, str]: Dictionary mapping matched acronyms to their expanded full names.
    """
    acronym_db = load_acronyms()
    resolved: dict[str, str] = {}

    if isinstance(input_data, str):
        tokens = set(re.findall(r"\b[A-Za-z0-9/\-]{2,10}\b", input_data))
    else:
        tokens = set(input_data)

    for token in tokens:
        clean_token = token.strip()
        if clean_token in acronym_db:
            resolved[clean_token] = acronym_db[clean_token]
        elif clean_token.upper() in acronym_db:
            resolved[clean_token] = acronym_db[clean_token.upper()]
        elif clean_token.lower() in acronym_db:
            resolved[clean_token] = acronym_db[clean_token.lower()]

    return resolved
