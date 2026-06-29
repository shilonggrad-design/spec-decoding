#!/usr/bin/env python3
"""
baseline.py — Day 2: Vanilla autoregressive decoding + xgrammar-constrained decoding.

Provides:
- load_models(): Load Qwen3.5-4B (target) + Qwen3.5-0.8B (draft) with shared tokenizer.
- generate_ar(): Greedy autoregressive generation (no grammar).
- generate_grammar_constrained(): Greedy AR generation with xgrammar bitmask constraints.
"""

from __future__ import annotations

import json
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import xgrammar as xgr

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
TARGET_MODEL_ID = "Qwen/Qwen3.5-4B"
DRAFT_MODEL_ID = "Qwen/Qwen3.5-0.8B"


def load_models(device: str = "auto") -> tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """Load target and draft models with a shared tokenizer.

    Returns:
        (target_model, draft_model, tokenizer)

    Raises:
        AssertionError: If vocab sizes do not match between models.
    """
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )

    # Both models must share the same vocabulary for speculative decoding to work.
    target_vocab = target_model.config.vocab_size
    draft_vocab = draft_model.config.vocab_size
    assert target_vocab == draft_vocab, (
        f"Vocab size mismatch: target={target_vocab}, draft={draft_vocab}"
    )

    target_model.eval()
    draft_model.eval()
    return target_model, draft_model, tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def popcount_mask(bitmask_row: torch.Tensor, vocab_size: int) -> int:
    """Count valid (set) bits in a packed int32 bitmask row.

    Each int32 element packs 32 bits — bit i globally corresponds to token i.
    We handle negative Python ints from unsigned→signed wrapping.

    Args:
        bitmask_row: 1-D tensor of int32 values.
        vocab_size:  Total vocabulary size (to sanity-check, not strictly needed).

    Returns:
        Number of bits set (valid tokens).
    """
    valid = 0
    for word in bitmask_row:
        bits = word.item()
        if bits < 0:
            bits += 1 << 32  # reinterpret as unsigned int32
        valid += bin(bits).count("1")
    return valid


# ---------------------------------------------------------------------------
# Vanilla AR generation (greedy)
# ---------------------------------------------------------------------------
def generate_ar(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_tokens: int = 256,
    device: str | None = None,
) -> dict[str, Any]:
    """Greedy autoregressive generation without grammar constraints.

    Returns:
        dict with keys: token_ids (list[int]), text (str), time_sec (float)
    """
    model_device = device or getattr(model, "device", "cpu")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_device)

    generated_ids: list[int] = []
    start = time.perf_counter()

    with torch.inference_mode():
        past_key_values = None
        current_ids = input_ids

        for _ in range(max_tokens):
            outputs = model(current_ids, past_key_values=past_key_values, use_cache=True)
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()

            if next_token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id)
            current_ids = torch.tensor([[next_token_id]], device=model_device)
            past_key_values = outputs.past_key_values

    elapsed = time.perf_counter() - start
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {"token_ids": generated_ids, "text": text, "time_sec": elapsed}


# ---------------------------------------------------------------------------
# Grammar-constrained AR generation (greedy)
# ---------------------------------------------------------------------------
def generate_grammar_constrained(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    schema_str: str,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Greedy autoregressive generation with xgrammar bitmask constraints.

    At each step:
      1. Forward pass → logits
      2. GrammarMatcher fills the token bitmask
      3. Apply bitmask (invalid → −inf) in-place on logits
      4. argmax → selected token
      5. Accept token into matcher → check termination

    Returns:
        dict with keys: token_ids, text, time_sec, density_trace (list[float])
    """
    vocab_size = model.config.vocab_size
    model_device = getattr(model, "device", "cpu")

    # --- Build grammar pipeline ---
    grammar = xgr.Grammar.from_json_schema(schema_str)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled_grammar)

    # Pre-allocate bitmask (batch=1)
    bitmask = xgr.allocate_token_bitmask(batch_size=1, vocab_size=vocab_size)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_device)
    generated_ids: list[int] = []
    density_trace: list[float] = []
    start = time.perf_counter()

    with torch.inference_mode():
        past_key_values = None
        current_ids = input_ids

        for _ in range(max_tokens):
            outputs = model(current_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)

            # Fill grammar bitmask and apply if needed
            need_apply = matcher.fill_next_token_bitmask(bitmask)
            if need_apply:
                xgr.apply_token_bitmask_inplace(logits, bitmask.to(model_device))
                # Record density (fraction of vocab that is valid)
                valid = popcount_mask(bitmask[0], vocab_size)
                density_trace.append(valid / vocab_size)
            else:
                density_trace.append(1.0)

            next_token_id = logits.argmax(dim=-1).item()
            accepted = matcher.accept_token(next_token_id)

            if not accepted:
                # Grammar rejected the token — shouldn't happen with proper masking,
                # but we break to avoid an infinite loop.
                print(f"[WARN] Grammar rejected token {next_token_id}")
                break

            if matcher.is_terminated() or next_token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id)
            current_ids = torch.tensor([[next_token_id]], device=model_device)
            past_key_values = outputs.past_key_values

    elapsed = time.perf_counter() - start
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "token_ids": generated_ids,
        "text": text,
        "time_sec": elapsed,
        "density_trace": density_trace,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Week 1 — Day 2: Baseline AR + Grammar-Constrained Generation")
    print("=" * 60)

    target_model, draft_model, tokenizer = load_models()
    print(f"[OK] Loaded {TARGET_MODEL_ID} + {DRAFT_MODEL_ID}")
    print(f"     Vocab size: {target_model.config.vocab_size}")

    # Load schema
    import os
    schema_path = os.path.join(os.path.dirname(__file__), "..", "schemas", "schema_simple.json")
    with open(schema_path) as f:
        schema_str = f.read()

    prompt = "Generate a person's information in JSON format."

    # --- Vanilla AR ---
    print("\n--- Vanilla AR ---")
    ar_result = generate_ar(target_model, tokenizer, prompt, max_tokens=256)
    print(f"Tokens: {len(ar_result['token_ids'])}")
    print(f"Time:  {ar_result['time_sec']:.3f}s")
    print(f"Output: {ar_result['text'][:200]}...")

    # --- Grammar-constrained AR ---
    print("\n--- Grammar-Constrained AR ---")
    gc_result = generate_grammar_constrained(
        target_model, tokenizer, prompt, schema_str, max_tokens=512
    )
    print(f"Tokens: {len(gc_result['token_ids'])}")
    print(f"Time:  {gc_result['time_sec']:.3f}s")
    print(f"Output: {gc_result['text'][:300]}...")
    avg_density = sum(gc_result["density_trace"]) / max(len(gc_result["density_trace"]), 1)
    print(f"Avg mask density: {avg_density:.4f} ({avg_density * 100:.1f}% of vocab)")


if __name__ == "__main__":
    main()
