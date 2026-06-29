#!/usr/bin/env python3
"""
speculator.py — Days 3–4: Speculative decoding with two configurations.

C1 (free spec):  No grammar anywhere. Draft and verify freely.
C2 (verify-only): Grammar bitmask applied only during target verification.
                   Draft is unconstrained.

Algorithm per round:
  1. Draft phase  — draft model autoregressively generates K tokens (greedy).
  2. Verify phase — target model single forward pass over [prefix + draft].
  3. Accept/reject (greedy) — compare argmax at each position.
  4. If all K accepted → bonus token from position prompt_len + K.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import xgrammar as xgr

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
TARGET_MODEL_ID = "Qwen/Qwen3.5-4B"
DRAFT_MODEL_ID = "Qwen/Qwen3.5-0.8B"

Config = Literal["C1", "C2"]


def load_models(device: str = "auto") -> tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """Load target (4B) and draft (0.8B) models with shared tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    assert target.config.vocab_size == draft.config.vocab_size, (
        f"Vocab mismatch: target={target.config.vocab_size}, draft={draft.config.vocab_size}"
    )
    target.eval()
    draft.eval()
    return target, draft, tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def popcount_mask(bitmask_row: torch.Tensor, vocab_size: int) -> int:
    """Count set bits in a packed int32 bitmask row."""
    valid = 0
    for word in bitmask_row:
        bits = word.item()
        if bits < 0:
            bits += 1 << 32
        valid += bin(bits).count("1")
    return valid


def build_grammar_pipeline(
    tokenizer: AutoTokenizer,
    vocab_size: int,
    schema_str: str,
) -> tuple[xgr.GrammarMatcher, torch.Tensor]:
    """Build the xgrammar pipeline: Grammar → TokenizerInfo → Compiler → Matcher.

    Also pre-allocates the bitmask tensor.

    Returns:
        (matcher, bitmask) — matcher is fresh/reset, bitmask is on CPU.
    """
    grammar = xgr.Grammar.from_json_schema(schema_str)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled = compiler.compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = xgr.allocate_token_bitmask(batch_size=1, vocab_size=vocab_size)
    return matcher, bitmask


def draft_k_tokens(
    draft_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    K: int,
    eos_token_id: int | None,
) -> list[int]:
    """Autoregressively generate up to K tokens with the draft model (greedy).

    Uses KV cache for efficiency.

    Args:
        draft_model: The draft (smaller) model.
        input_ids: Current prefix tensor, shape (1, seq_len).
        K: Maximum number of tokens to draft.
        eos_token_id: Token ID for end-of-sequence.

    Returns:
        List of drafted token IDs (length 0..K).
    """
    draft_model_device = getattr(draft_model, "device", "cpu")
    drafted: list[int] = []
    past_key_values = None
    current = input_ids.to(draft_model_device)

    with torch.inference_mode():
        for _ in range(K):
            outputs = draft_model(current, past_key_values=past_key_values, use_cache=True)
            next_id = outputs.logits[:, -1, :].argmax(dim=-1).item()

            if next_id == eos_token_id:
                break

            drafted.append(next_id)
            current = torch.tensor([[next_id]], device=draft_model_device)
            past_key_values = outputs.past_key_values

    return drafted


# ---------------------------------------------------------------------------
# Core: Speculative decoding
# ---------------------------------------------------------------------------
def speculative_decode(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_tokens: list[int],
    K: int = 5,
    config: Config = "C1",
    schema_str: str | None = None,
    max_tokens: int = 256,
) -> dict[str, Any]:
    """Run speculative decoding with the given configuration.

    Args:
        target_model: Large target model (Qwen3.5-4B).
        draft_model: Small draft model (Qwen3.5-0.8B).
        tokenizer: Shared HuggingFace tokenizer.
        prompt_tokens: Tokenized prompt (list of ints).
        K: Speculation width — number of tokens to draft per round.
        config: "C1" (free spec) or "C2" (verify-only grammar).
        schema_str: JSON schema string (required for C2).
        max_tokens: Maximum tokens to generate (excluding prompt).

    Returns:
        dict with keys:
            token_ids (list[int]),
            text (str),
            accepted_count (int),
            drafted_count (int),
            acceptance_rate (float),
            time_sec (float),
            density_trace (list[float], C2 only),
            rounds (int)
    """
    assert config in ("C1", "C2"), f"Invalid config: {config}"
    if config == "C2":
        assert schema_str is not None, "C2 requires a schema_str"

    vocab_size = target_model.config.vocab_size
    target_device = getattr(target_model, "device", "cpu")
    eos_token_id = tokenizer.eos_token_id

    # Build grammar pipeline for C2
    matcher: xgr.GrammarMatcher | None = None
    bitmask: torch.Tensor | None = None
    if config == "C2":
        matcher, bitmask = build_grammar_pipeline(tokenizer, vocab_size, schema_str)

    generated: list[int] = []
    total_accepted = 0
    total_drafted = 0
    density_trace: list[float] = []
    rounds = 0
    start = time.perf_counter()

    with torch.inference_mode():
        while len(generated) < max_tokens:
            # Current full prefix
            prefix = torch.tensor(
                [prompt_tokens + generated], dtype=torch.long, device=target_device
            )
            prompt_len = len(prompt_tokens)

            # ---- 1. Draft phase ----
            draft_tokens = draft_k_tokens(draft_model, prefix, K, eos_token_id)
            if not draft_tokens:
                break  # Draft produced EOS immediately
            total_drafted += len(draft_tokens)

            # ---- 2. Verify phase ----
            # Full forward pass over [prefix + draft_tokens] — no KV cache for target (Week 1 simplicity)
            verify_input = torch.tensor(
                [prompt_tokens + generated + draft_tokens],
                dtype=torch.long,
                device=target_device,
            )
            verify_outputs = target_model(verify_input)
            verify_logits = verify_outputs.logits[0]  # (seq_len, vocab_size)

            # Debug: log shapes
            if rounds == 0:
                print(f"[DEBUG] verify_input len={verify_input.shape[1]}, logits shape={verify_logits.shape[0]}")
                print(f"[DEBUG] prompt_len={prompt_len}, len(generated)={len(generated)}, n_draft={n_draft}")

            # ---- 3. Accept / Reject (greedy) ----
            accepted_this_round = 0
            n_draft = len(draft_tokens)

            for i in range(n_draft):
                # logits[pos] predicts token at pos+1
                # prefix = prompt_tokens + generated (length = prompt_len + len(generated))
                # logits at prefix_end - 1 predicts first draft token
                pos = prompt_len + len(generated) - 1 + i
                if pos >= verify_logits.shape[0]:
                    # Out of bounds — skip remaining draft tokens
                    break
                position_logits = verify_logits[pos]  # (vocab_size,)

                # Apply grammar mask for C2 before argmax
                if config == "C2" and matcher is not None and bitmask is not None:
                    need_apply = matcher.fill_next_token_bitmask(bitmask)
                    if need_apply:
                        xgr.apply_token_bitmask_inplace(
                            position_logits.unsqueeze(0), bitmask.to(target_device)
                        )
                        position_logits = position_logits.unsqueeze(0)[0]
                        valid = popcount_mask(bitmask[0], vocab_size)
                        density_trace.append(valid / vocab_size)
                    else:
                        density_trace.append(1.0)

                target_predicted = position_logits.argmax().item()

                if target_predicted == draft_tokens[i]:
                    # Accept
                    generated.append(draft_tokens[i])
                    accepted_this_round += 1
                    total_accepted += 1

                    # Tell grammar matcher about accepted token
                    if config == "C2" and matcher is not None:
                        accepted = matcher.accept_token(draft_tokens[i])
                        if not accepted:
                            # Grammar mismatch — accept anyway (mask should prevent this)
                            print(f"[WARN] Grammar rejected accepted token {draft_tokens[i]}")
                        if matcher.is_terminated():
                            break
                else:
                    # Reject — take target's correction
                    generated.append(target_predicted)
                    if config == "C2" and matcher is not None:
                        matcher.accept_token(target_predicted)
                    break

            # If all K accepted AND not terminated, grab bonus token
            if accepted_this_round == n_draft:
                # logits at last position of [prefix + draft] predicts next token
                bonus_pos = prompt_len + len(generated) + n_draft - 1
                if bonus_pos < verify_logits.shape[0]:
                    bonus_logits = verify_logits[bonus_pos].clone()

                    if config == "C2" and matcher is not None and bitmask is not None:
                        need_apply = matcher.fill_next_token_bitmask(bitmask)
                        if need_apply:
                            xgr.apply_token_bitmask_inplace(
                                bonus_logits.unsqueeze(0), bitmask.to(target_device)
                            )
                            valid = popcount_mask(bitmask[0], vocab_size)
                            density_trace.append(valid / vocab_size)
                        else:
                            density_trace.append(1.0)

                    bonus_token = bonus_logits.argmax().item()

                    if bonus_token == eos_token_id:
                        break

                    generated.append(bonus_token)
                    if config == "C2" and matcher is not None:
                        matcher.accept_token(bonus_token)
                        if matcher.is_terminated():
                            break

            rounds += 1

            if eos_token_id in generated:
                # Trim at EOS
                eos_idx = generated.index(eos_token_id)
                generated = generated[:eos_idx]
                break

    elapsed = time.perf_counter() - start
    acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0
    text = tokenizer.decode(generated, skip_special_tokens=True)

    result: dict[str, Any] = {
        "token_ids": generated,
        "text": text,
        "accepted_count": total_accepted,
        "drafted_count": total_drafted,
        "acceptance_rate": acceptance_rate,
        "time_sec": elapsed,
        "rounds": rounds,
    }
    if density_trace:
        result["density_trace"] = density_trace

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import os

    print("=" * 60)
    print("Week 1 — Days 3–4: Speculative Decoding (C1 + C2)")
    print("=" * 60)

    target_model, draft_model, tokenizer = load_models()
    print(f"[OK] Loaded {TARGET_MODEL_ID} + {DRAFT_MODEL_ID}")

    # Load schema
    schema_path = os.path.join(os.path.dirname(__file__), "..", "schemas", "schema_simple.json")
    with open(schema_path) as f:
        schema_str = f.read()

    prompt = "Generate a person's information in JSON format."
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)

    print(f"\nPrompt: {prompt}")
    print(f"Prompt tokens: {len(prompt_tokens)}")

    # --- C1: Free speculative decoding ---
    print("\n--- C1: Free Speculative Decoding ---")
    c1_result = speculative_decode(
        target_model, draft_model, tokenizer, prompt_tokens,
        K=5, config="C1", max_tokens=256,
    )
    print(f"Tokens generated: {len(c1_result['token_ids'])}")
    print(f"Acceptance rate:  {c1_result['acceptance_rate']:.2%} "
          f"({c1_result['accepted_count']}/{c1_result['drafted_count']})")
    print(f"Time: {c1_result['time_sec']:.3f}s ({len(c1_result['token_ids']) / max(c1_result['time_sec'], 0.001):.1f} tok/s)")
    print(f"Output: {c1_result['text'][:300]}...")

    # --- C2: Verify-only grammar ---
    print("\n--- C2: Verify-Only Grammar Speculative Decoding ---")
    c2_result = speculative_decode(
        target_model, draft_model, tokenizer, prompt_tokens,
        K=5, config="C2", schema_str=schema_str, max_tokens=256,
    )
    print(f"Tokens generated: {len(c2_result['token_ids'])}")
    print(f"Acceptance rate:  {c2_result['acceptance_rate']:.2%} "
          f"({c2_result['accepted_count']}/{c2_result['drafted_count']})")
    print(f"Time: {c2_result['time_sec']:.3f}s ({len(c2_result['token_ids']) / max(c2_result['time_sec'], 0.001):.1f} tok/s)")
    print(f"Output: {c2_result['text'][:300]}...")

    if "density_trace" in c2_result and c2_result["density_trace"]:
        avg_d = sum(c2_result["density_trace"]) / len(c2_result["density_trace"])
        print(f"Avg mask density: {avg_d:.4f} ({avg_d * 100:.1f}% of vocab)")

    # --- Comparison ---
    print("\n--- Comparison ---")
    print(f"{'Metric':<20} {'C1':>12} {'C2':>12}")
    print("-" * 44)
    print(f"{'Tokens':<20} {len(c1_result['token_ids']):>12} {len(c2_result['token_ids']):>12}")
    print(f"{'Time (s)':<20} {c1_result['time_sec']:>12.3f} {c2_result['time_sec']:>12.3f}")
    print(f"{'Accept rate':<20} {c1_result['acceptance_rate']:>12.2%} {c2_result['acceptance_rate']:>12.2%}")


if __name__ == "__main__":
    main()
