#!/usr/bin/env python3
"""
speculator.py — Speculative decoding with four configurations.

C1 (free spec):     No grammar anywhere. Draft and verify freely.
C2 (verify-only):   Grammar bitmask applied only during target verification.
                    Draft is unconstrained.
C3 (grammar-guided): Grammar bitmask on BOTH draft and target. Draft generates
                     grammar-valid tokens, target verifies. After draft, matcher
                     rolls back to pre-draft state so verify phase reconstructs
                     correct grammar path from actual accepted tokens.
C4 (GrammarSD full): C3 + adaptive K driven by grammar mask density. Low density
                     → large K (speculate aggressively); high density → small K
                     (don't waste compute). Based on SpecDec++ threshold policy.

Algorithm per round:
  1. Draft phase  — draft model autoregressively generates K tokens (greedy).
                    C3/C4: grammar mask applied per token, matcher advances, then rollback.
                    C4: K computed from current position density before drafting.
  2. Verify phase — target model single forward pass over [prefix + draft].
                    C2/C3/C4: grammar mask applied on target logits per position.
  3. Accept/reject (greedy) — compare argmax at each position.
  4. If all K accepted → bonus token from logits at last position.
"""

from __future__ import annotations

import os
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

Config = Literal["C1", "C2", "C3", "C4"]


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
    matcher: xgr.GrammarMatcher | None = None,
    bitmask: torch.Tensor | None = None,
    vocab_size: int | None = None,
) -> list[int]:
    """Autoregressively generate up to K tokens with the draft model (greedy).

    If matcher is provided (C3), applies grammar mask before argmax and
    advances grammar state per token. Caller must rollback() after verification.

    Uses KV cache for efficiency.
    """
    draft_model_device = getattr(draft_model, "device", "cpu")
    drafted: list[int] = []
    past_key_values = None
    current = input_ids.to(draft_model_device)

    with torch.inference_mode():
        for _ in range(K):
            outputs = draft_model(current, past_key_values=past_key_values, use_cache=True)
            next_logits = outputs.logits[:, -1, :].clone()

            if matcher is not None and bitmask is not None:
                # C3: grammar-guided draft
                need_apply = matcher.fill_next_token_bitmask(bitmask)
                if need_apply:
                    xgr.apply_token_bitmask_inplace(
                        next_logits, bitmask.to(draft_model_device)
                    )

            next_id = next_logits.argmax(dim=-1).item()

            if next_id == eos_token_id:
                break

            drafted.append(next_id)
            current = torch.tensor([[next_id]], device=draft_model_device)
            past_key_values = outputs.past_key_values

            if matcher is not None:
                matcher.accept_token(next_id)
                if matcher.is_terminated():
                    break

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
        config: "C1" (free spec), "C2" (verify-only grammar), or "C3" (grammar-guided draft + verify).
        schema_str: JSON schema string (required for C2 and C3).
        max_tokens: Maximum tokens to generate (excluding prompt).

    Returns:
        dict with keys: token_ids, text, accepted_count, drafted_count,
        acceptance_rate, time_sec, density_trace (C2/C3/C4 only),
        k_trace (C4 only), rounds
    """
    assert config in ("C1", "C2", "C3", "C4"), f"Invalid config: {config}"
    if config in ("C2", "C3", "C4"):
        assert schema_str is not None, f"{config} requires a schema_str"

    vocab_size = target_model.config.vocab_size
    target_device = getattr(target_model, "device", "cpu")
    eos_token_id = tokenizer.eos_token_id

    # Build grammar pipeline for C2/C3/C4
    matcher: xgr.GrammarMatcher | None = None
    bitmask: torch.Tensor | None = None
    if config in ("C2", "C3", "C4"):
        matcher, bitmask = build_grammar_pipeline(tokenizer, vocab_size, schema_str)

    generated: list[int] = []
    total_accepted = 0
    total_drafted = 0
    density_trace: list[float] = []
    k_trace: list[tuple[float, int]] = []  # (density, K) per round, C4 only
    rounds = 0
    start = time.perf_counter()

    with torch.inference_mode():
        while len(generated) < max_tokens:
            prompt_len = len(prompt_tokens)
            prefix_len = prompt_len + len(generated)

            # ---- 1. Draft phase ----
            prefix = torch.tensor(
                [prompt_tokens + generated], dtype=torch.long, device=target_device
            )

            # C4: compute adaptive K from current position density
            current_K = K  # default fixed K
            if config == "C4" and matcher is not None and bitmask is not None:
                # Import with fallback for different working directories (Colab vs local)
                try:
                    from src.adaptive_k import compute_density, adaptive_K
                except ImportError:
                    try:
                        from adaptive_k import compute_density, adaptive_K
                    except ImportError:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location(
                            "adaptive_k",
                            os.path.join(os.path.dirname(__file__), "adaptive_k.py"),
                        )
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        compute_density = mod.compute_density
                        adaptive_K = mod.adaptive_K
                need_apply = matcher.fill_next_token_bitmask(bitmask)
                if need_apply:
                    density = compute_density(bitmask[0], vocab_size)
                    current_K = adaptive_K(density)
                else:
                    density = 1.0
                    current_K = K
                k_trace.append((density, current_K))

            if config in ("C3", "C4") and matcher is not None and bitmask is not None:
                draft_tokens = draft_k_tokens(
                    draft_model, prefix, current_K, eos_token_id,
                    matcher=matcher, bitmask=bitmask, vocab_size=vocab_size,
                )
                # Rollback matcher to pre-draft state for verify phase
                if len(draft_tokens) > 0:
                    matcher.rollback(len(draft_tokens))
            else:
                draft_tokens = draft_k_tokens(draft_model, prefix, current_K, eos_token_id)
            if not draft_tokens:
                break  # Draft produced EOS immediately
            n_draft = len(draft_tokens)
            # total_drafted updated after verification (C2 early-term fairness)
            # For C1, all drafted tokens will be verified, so defer is safe.

            # ---- 2. Verify phase ----
            # Full forward pass: input = [prompt + generated + draft_tokens]
            verify_input = torch.tensor(
                [prompt_tokens + generated + draft_tokens],
                dtype=torch.long,
                device=target_device,
            )
            verify_outputs = target_model(verify_input)
            verify_logits = verify_outputs.logits[0]  # (seq_len, vocab_size)
            # verify_logits has same length as verify_input
            # logits[i] predicts token at position i+1
            # So logits[prefix_len - 1] predicts the first draft token
            # logits[prefix_len - 1 + i] predicts draft_token[i]

            # ---- 3. Accept / Reject (greedy) ----
            accepted_this_round = 0
            rejected = False
            verified_this_round = 0  # only count tokens actually verified

            for i in range(n_draft):
                pos = prefix_len - 1 + i  # logits index for predicting draft_token[i]
                if pos >= verify_logits.shape[0]:
                    break  # safety bound
                verified_this_round += 1
                position_logits = verify_logits[pos].clone()  # (vocab_size,)

                # Apply grammar mask for C2/C3/C4 before argmax
                if config in ("C2", "C3", "C4") and matcher is not None and bitmask is not None:
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
                    if config in ("C2", "C3", "C4") and matcher is not None:
                        accepted = matcher.accept_token(draft_tokens[i])
                        if not accepted:
                            print(f"[WARN] Grammar rejected accepted token {draft_tokens[i]}")
                        if matcher.is_terminated():
                            break
                else:
                    # Reject — take target's correction
                    generated.append(target_predicted)
                    if config in ("C2", "C3", "C4") and matcher is not None:
                        matcher.accept_token(target_predicted)
                    rejected = True
                    break

            # ---- 4. Bonus token (if all K accepted and not terminated) ----
            total_drafted += verified_this_round  # only tokens actually checked

            if accepted_this_round == n_draft and not rejected:
                # All draft tokens accepted. The last logits position predicts the next token.
                bonus_pos = prefix_len + n_draft - 1  # = len(verify_input) - 1
                if bonus_pos < verify_logits.shape[0]:
                    bonus_logits = verify_logits[bonus_pos].clone()

                    if config in ("C2", "C3", "C4") and matcher is not None and bitmask is not None:
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
                    if config in ("C2", "C3", "C4") and matcher is not None:
                        matcher.accept_token(bonus_token)
                        if matcher.is_terminated():
                            break

            rounds += 1

            if eos_token_id in generated:
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
    if k_trace:
        result["k_trace"] = k_trace

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import os

    print("=" * 60)
    print("Speculative Decoding: C1 + C2 + C3 + C4 (full ablation)")
    print("=" * 60)

    target_model, draft_model, tokenizer = load_models()
    print(f"[OK] Loaded {TARGET_MODEL_ID} + {DRAFT_MODEL_ID}")

    # Load schema
    schema_path = os.path.join(os.path.dirname(__file__), "..", "schemas", "schema_tool_call.json")
    with open(schema_path) as f:
        schema_str = f.read()

    prompt = "Call a function to search for AI news."
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)

    print(f"\nPrompt: {prompt}")
    print(f"Schema: tool_call")
    print(f"Prompt tokens: {len(prompt_tokens)}")

    results = {}
    for cfg in ("C1", "C2", "C3", "C4"):
        print(f"\n--- {cfg} ---")
        result = speculative_decode(
            target_model, draft_model, tokenizer, prompt_tokens,
            K=5, config=cfg,
            schema_str=schema_str if cfg != "C1" else None,
            max_tokens=256,
        )
        results[cfg] = result
        print(f"Tokens generated: {len(result['token_ids'])}")
        print(f"Acceptance rate:  {result['acceptance_rate']:.2%} "
              f"({result['accepted_count']}/{result['drafted_count']})")
        print(f"Time: {result['time_sec']:.3f}s ({len(result['token_ids']) / max(result['time_sec'], 0.001):.1f} tok/s)")
        print(f"Output: {result['text'][:300]}...")
        if "k_trace" in result:
            print(f"K trace: {result['k_trace']}")

    # --- Comparison ---
    print("\n--- Comparison ---")
    print(f"{'Metric':<20} {'C1':>10} {'C2':>10} {'C3':>10} {'C4':>10}")
    print("-" * 60)
    print(f"{'Tokens':<20} {len(results['C1']['token_ids']):>10} {len(results['C2']['token_ids']):>10} {len(results['C3']['token_ids']):>10} {len(results['C4']['token_ids']):>10}")
    print(f"{'Time (s)':<20} {results['C1']['time_sec']:>10.3f} {results['C2']['time_sec']:>10.3f} {results['C3']['time_sec']:>10.3f} {results['C4']['time_sec']:>10.3f}")
    print(f"{'Accept rate':<20} {results['C1']['acceptance_rate']:>10.2%} {results['C2']['acceptance_rate']:>10.2%} {results['C3']['acceptance_rate']:>10.2%} {results['C4']['acceptance_rate']:>10.2%}")


if __name__ == "__main__":
    main()
