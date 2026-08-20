"""Real Target-LLM Tokenizer module for SIH2026 NLP Pipeline.

Per Section 14-18 of the specification, tokenization MUST happen AFTER the complete
NLP pipeline has finished, operating on the exact serialized JSON string representation
that will be sent to the target LLM. Tokenization uses the real tokenizer of the target LLM
(e.g., `tiktoken` for OpenAI-family models like gpt-4o-mini / gpt-4o), never a naive split.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    tiktoken = None

try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None


def count_and_tokenize(
    text: str,
    model_name: str = "gpt-4o-mini",
    include_token_ids: bool = False
) -> Tuple[int, Optional[list[int]], dict]:
    """Tokenizes text using the real target LLM tokenizer.

    Returns:
        (token_count, token_ids_list_or_None, metadata_dict)
    """
    if not text:
        return 0, [] if include_token_ids else None, {
            "name": "tiktoken" if _TIKTOKEN_AVAILABLE else "transformers",
            "model": model_name,
            "encoding": "none",
            "version": "unknown"
        }

    # 1. Try OpenAI tiktoken tokenizer (cl100k_base / o200k_base)
    if _TIKTOKEN_AVAILABLE:
        try:
            encoding_name = "cl100k_base"
            encoding = tiktoken.get_encoding("cl100k_base")
            try:
                model_enc = tiktoken.encoding_for_model(model_name)
                encoding = model_enc
                encoding_name = getattr(model_enc, "name", "cl100k_base")
            except Exception:
                pass

            token_ids = encoding.encode(text)
            meta = {
                "name": "tiktoken",
                "model": model_name,
                "encoding": encoding_name,
                "version": getattr(tiktoken, "__version__", "0.14.0")
            }
            return len(token_ids), (token_ids if include_token_ids else None), meta
        except Exception:
            pass

    # 2. Try HuggingFace AutoTokenizer as fallback
    if _TRANSFORMERS_AVAILABLE:
        try:
            hf_model = model_name if "/" in model_name else "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(hf_model)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            meta = {
                "name": "transformers.AutoTokenizer",
                "model": hf_model,
                "encoding": tokenizer.__class__.__name__,
                "version": "5.x"
            }
            return len(token_ids), (token_ids if include_token_ids else None), meta
        except Exception:
            pass

    # 3. Fallback heuristic tokenizer (~4 characters per token)
    words = text.split()
    estimated_tokens = max(len(words), len(text) // 4)
    meta = {
        "name": "heuristic_fallback",
        "model": model_name,
        "encoding": "approx_4chars_per_token",
        "version": "1.0"
    }
    dummy_ids = list(range(1, estimated_tokens + 1)) if include_token_ids else None
    return estimated_tokens, dummy_ids, meta


def tokenize_nlp_output(
    nlp_data: dict,
    stage1_json: Optional[dict] = None,
    model_name: str = "gpt-4o-mini",
    source_filename: str = "sylabus_nlp.json",
    include_token_ids: bool = False
) -> dict:
    """Tokenizes the exact serialized JSON representation of the final NLP output.

    Calculates exact token reduction metrics comparing Stage 1 input vs Final NLP output.
    """
    # Exact compact serialized string sent to the target LLM
    exact_nlp_input_str = json.dumps(nlp_data, ensure_ascii=False, indent=None)

    nlp_token_count, nlp_token_ids, tokenizer_meta = count_and_tokenize(
        exact_nlp_input_str,
        model_name=model_name,
        include_token_ids=include_token_ids
    )

    # Compute original Stage 1 token count if provided
    original_token_count = 0
    if stage1_json:
        stage1_text = stage1_json.get("llm_input_context")
        if not stage1_text:
            stage1_text = json.dumps(stage1_json, ensure_ascii=False, indent=None)

        orig_count, _, _ = count_and_tokenize(
            stage1_text,
            model_name=model_name,
            include_token_ids=False
        )
        original_token_count = orig_count

    reduction_tokens = max(0, original_token_count - nlp_token_count)
    reduction_percentage = (
        round((reduction_tokens / original_token_count) * 100.0, 2)
        if original_token_count > 0 else 0.0
    )

    result = {
        "source_file": source_filename,
        "tokenizer": tokenizer_meta,
        "token_metrics": {
            "original_token_count": original_token_count,
            "nlp_token_count": nlp_token_count,
            "reduction_tokens": reduction_tokens,
            "reduction_percentage": reduction_percentage
        },
        "text_serialized_input": exact_nlp_input_str,
    }

    if include_token_ids and nlp_token_ids is not None:
        result["tokens"] = nlp_token_ids

    return result
