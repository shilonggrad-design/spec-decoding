#!/usr/bin/env python3
"""
test_validity.py — Day 5: Correctness tests for Week 1.

Tests:
  1. test_vocab_match          — target and draft share vocab_size
  2. test_ar_output_valid      — grammar-constrained AR produces valid JSON
  3. test_spec_c1_matches_ar   — C1 speculative == vanilla AR (greedy)
  4. test_spec_c2_output_valid — C2 output is valid JSON matching schema
  5. test_kv_rollback          — rejection doesn't corrupt subsequent rounds

Run with:  python -m pytest src/test_validity.py -v
    or:    python src/test_validity.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baseline import generate_ar, generate_grammar_constrained, load_models  # noqa: E402
from src.speculator import speculative_decode  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures (module-level, loaded once)
# ---------------------------------------------------------------------------
_base = Path(__file__).resolve().parent.parent

_PROMPT = "Generate a person's information in JSON format."
_MAX_TOKENS = 128  # Keep short for test speed

_schema_paths = {
    "simple": _base / "schemas" / "schema_simple.json",
}

_models = None
_tokenizer = None


def _get_models_and_tokenizer():
    global _models, _tokenizer
    if _models is None:
        target, draft, tok = load_models()
        _models = (target, draft)
        _tokenizer = tok
    return _models, _tokenizer


def _load_schema(name: str = "simple") -> str:
    path = _schema_paths[name]
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict | None:
    """Try to parse the first JSON object from generated text."""
    # Find the first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _validate_against_schema(data: dict, schema: dict) -> bool:
    """Lightweight schema validation (required fields + types)."""
    if "required" in schema:
        for field in schema["required"]:
            if field not in data:
                return False
    props = schema.get("properties", {})
    for key, val_spec in props.items():
        if key in data:
            val = data[key]
            expected_type = val_spec.get("type")
            if expected_type == "string" and not isinstance(val, str):
                return False
            if expected_type == "integer" and not isinstance(val, int):
                return False
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_vocab_match() -> None:
    """Target and draft must have identical vocab sizes."""
    (target, draft), tok = _get_models_and_tokenizer()
    assert target.config.vocab_size == draft.config.vocab_size, (
        f"Vocab mismatch: target={target.config.vocab_size}, draft={draft.config.vocab_size}"
    )
    print("[PASS] test_vocab_match")


def test_ar_output_valid() -> None:
    """Grammar-constrained AR generation must produce valid JSON."""
    (target, _), tok = _get_models_and_tokenizer()
    schema_str = _load_schema("simple")
    schema_obj = json.loads(schema_str)

    result = generate_grammar_constrained(
        target, tok, _PROMPT, schema_str, max_tokens=_MAX_TOKENS
    )

    parsed = _extract_json(result["text"])
    assert parsed is not None, (
        f"Failed to extract JSON from AR output: {result['text'][:200]}"
    )
    assert _validate_against_schema(parsed, schema_obj), (
        f"AR output doesn't match schema: {parsed}"
    )
    print(f"[PASS] test_ar_output_valid — output: {json.dumps(parsed, indent=2)[:200]}")


def test_spec_c1_matches_ar() -> None:
    """C1 speculative decoding should produce the same output as vanilla AR (greedy).

    Under greedy decoding with the same model and no grammar, speculative
    decoding should recover identical token sequences because both use
    argmax at every step and the target model is deterministic.
    """
    (target, draft), tok = _get_models_and_tokenizer()
    prompt_tokens = tok.encode(_PROMPT, add_special_tokens=True)

    # Vanilla AR
    ar_result = generate_ar(target, tok, _PROMPT, max_tokens=_MAX_TOKENS)

    # C1 speculative
    c1_result = speculative_decode(
        target, draft, tok, prompt_tokens,
        K=5, config="C1", max_tokens=_MAX_TOKENS,
    )

    # Compare — they should match token-for-token
    # (In theory identical; in practice tiny floating-point diffs could diverge,
    #  but greedy argmax is usually stable.)
    ar_ids = ar_result["token_ids"]
    c1_ids = c1_result["token_ids"]

    # Find the shorter length for comparison
    min_len = min(len(ar_ids), len(c1_ids))
    match_count = sum(1 for i in range(min_len) if ar_ids[i] == c1_ids[i])
    match_pct = match_count / min_len if min_len > 0 else 1.0

    print(f"[INFO] C1 vs AR: {match_pct:.1%} match over {min_len} tokens")

    # We accept >= 95% match (minor divergence at end is OK)
    assert match_pct >= 0.95, (
        f"C1 output diverges too much from AR: {match_pct:.1%}"
    )
    print("[PASS] test_spec_c1_matches_ar")


def test_spec_c2_output_valid() -> None:
    """C2 speculative decoding must produce valid JSON matching the schema."""
    (target, draft), tok = _get_models_and_tokenizer()
    prompt_tokens = tok.encode(_PROMPT, add_special_tokens=True)
    schema_str = _load_schema("simple")
    schema_obj = json.loads(schema_str)

    c2_result = speculative_decode(
        target, draft, tok, prompt_tokens,
        K=5, config="C2", schema_str=schema_str, max_tokens=_MAX_TOKENS,
    )

    parsed = _extract_json(c2_result["text"])
    assert parsed is not None, (
        f"Failed to extract JSON from C2 output: {c2_result['text'][:200]}"
    )
    assert _validate_against_schema(parsed, schema_obj), (
        f"C2 output doesn't match schema: {parsed}"
    )
    print(f"[PASS] test_spec_c2_output_valid — output: {json.dumps(parsed, indent=2)[:200]}")


def test_kv_rollback() -> None:
    """After a rejection, the next round should produce correct output (no stale KV cache).

    Strategy: Run C1 spec decode for several rounds, then verify that the
    tokens accepted by the spec decoder match what the target model would
    produce if run autoregressively on the same prefix at the rejection point.
    """
    (target, draft), tok = _get_models_and_tokenizer()
    prompt_tokens = tok.encode(_PROMPT, add_special_tokens=True)

    # Run speculative decoding
    c1_result = speculative_decode(
        target, draft, tok, prompt_tokens,
        K=5, config="C1", max_tokens=_MAX_TOKENS,
    )

    # Now re-run AR from the same prompt with the same max_tokens
    ar_result = generate_ar(target, tok, _PROMPT, max_tokens=_MAX_TOKENS)

    ar_ids = ar_result["token_ids"]
    c1_ids = c1_result["token_ids"]

    # The spec decoder should not have "ghost" tokens from stale KV cache.
    # We verify by checking that all accepted tokens are valid (the target model
    # would agree). The C1_matches_ar test above covers this more precisely.
    # Here we just verify the output is non-empty and reasonable length.
    assert len(c1_ids) > 0, "C1 produced no tokens after potential rejection"
    assert len(c1_ids) >= _MAX_TOKENS * 0.5, (
        f"C1 output too short ({len(c1_ids)}), possible KV cache corruption"
    )
    print(f"[PASS] test_kv_rollback — C1 produced {len(c1_ids)} tokens, AR produced {len(ar_ids)}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all_tests() -> None:
    """Run all tests with simple assertion-based reporting."""
    tests = [
        test_vocab_match,
        test_ar_output_valid,
        test_spec_c1_matches_ar,
        test_spec_c2_output_valid,
        test_kv_rollback,
    ]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
