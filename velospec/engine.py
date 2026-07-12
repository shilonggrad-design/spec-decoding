"""
engine.py — VeloSpec engine: grammar-aware adaptive speculative decoding.

Four configurations (ablation study):
  C1 (free spec):     No grammar anywhere. Draft and verify freely.
  C2 (verify-only):   Grammar bitmask applied only during target verification.
  C3 (grammar-guided): Grammar on BOTH draft and target. Draft generates
                       grammar-valid tokens, target verifies.
  C4 (VeloSpec full):  C3 + adaptive K driven by grammar mask density.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import xgrammar as xgr

from velospec.adaptive_k import compute_density, adaptive_K

Config = Literal["C1", "C2", "C3", "C4"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    """Result from a single generation call."""
    text: str
    token_ids: list[int]
    accepted_count: int
    drafted_count: int
    acceptance_rate: float
    time_sec: float
    tokens_per_sec: float
    rounds: int
    config: str
    density_trace: list[float] = field(default_factory=list)
    k_trace: list[tuple[float, int]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"GenerationResult(text={self.text[:80]!r}..., "
            f"acceptance={self.acceptance_rate:.2%}, "
            f"tokens={len(self.token_ids)}, "
            f"tps={self.tokens_per_sec:.1f})"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class VeloSpec:
    """Grammar-aware adaptive speculative decoding engine.

    Args:
        target_model: HuggingFace model ID for the large target model.
        draft_model: HuggingFace model ID for the small draft model.
        config: One of "C1" (free spec), "C2" (verify-only grammar),
                "C3" (grammar-guided draft), "C4" (C3 + adaptive K).
        K: Speculation width for C1/C2/C3. Serves as K_MAX for C4.
        device: "auto", "cuda", or "cpu".
        dtype: torch.float16 or torch.float32 (default: auto).

    Example:
        >>> engine = VeloSpec("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-0.8B", config="C4")
        >>> result = engine.generate("Hello", schema={"type": "object"})
        >>> print(result.text)
    """

    def __init__(
        self,
        target_model: str,
        draft_model: str,
        config: Config = "C4",
        K: int = 5,
        device: str = "auto",
        dtype: torch.dtype | None = None,
        backend: str = "auto",
    ) -> None:
        assert config in ("C1", "C2", "C3", "C4"), f"Invalid config: {config}"
        self.target_id = target_model
        self.draft_id = draft_model
        self.config = config
        self.K = K
        self._device = "cuda" if device == "auto" and torch.cuda.is_available() else device

        if dtype is None:
            self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            self._dtype = dtype

        # Backend: "auto" → prefer Triton, "cuda" → xgrammar+PyTorch fallback
        if backend == "auto":
            try:
                from velospec.triton.fused_logit_processor import is_available
                self._backend = "triton" if is_available() else "cuda"
            except ImportError:
                self._backend = "cuda"
        else:
            self._backend = backend

        # Lazy-loaded
        self._target: AutoModelForCausalLM | None = None
        self._draft: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._vocab_size: int | None = None
        self._eos_token_id: int | None = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._target is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.target_id, trust_remote_code=True
        )
        self._target = AutoModelForCausalLM.from_pretrained(
            self.target_id, torch_dtype=self._dtype, device_map=self._device,
            trust_remote_code=True,
        )
        self._draft = AutoModelForCausalLM.from_pretrained(
            self.draft_id, torch_dtype=self._dtype, device_map=self._device,
            trust_remote_code=True,
        )

        assert self._target.config.vocab_size == self._draft.config.vocab_size, (
            f"Vocab mismatch: target={self._target.config.vocab_size}, "
            f"draft={self._draft.config.vocab_size}"
        )

        self._vocab_size = self._target.config.vocab_size
        self._eos_token_id = self._tokenizer.eos_token_id
        self._target.eval()
        self._draft.eval()

    @property
    def device(self) -> str:
        if self._target is not None:
            return str(getattr(self._target, "device", "cpu"))
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Schema normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_schema(schema: dict | list | str | None) -> str | None:
        """Accept dict, JSON string, or file path → return JSON string."""
        if schema is None:
            return None
        if isinstance(schema, (dict, list)):
            return json.dumps(schema)
        if isinstance(schema, str):
            # Could be a file path or raw JSON string
            if os.path.exists(schema):
                with open(schema) as f:
                    return f.read()
            return schema  # assume raw JSON
        raise TypeError(f"schema must be dict/str/None, got {type(schema)}")

    # ------------------------------------------------------------------
    # Grammar pipeline
    # ------------------------------------------------------------------
    def _build_grammar_pipeline(self, schema_str: str) -> tuple[xgr.GrammarMatcher, torch.Tensor]:
        grammar = xgr.Grammar.from_json_schema(schema_str)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self._tokenizer, vocab_size=self._vocab_size
        )
        compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = compiler.compile_grammar(grammar)
        matcher = xgr.GrammarMatcher(compiled)
        bitmask = xgr.allocate_token_bitmask(
            batch_size=1, vocab_size=self._vocab_size
        )
        return matcher, bitmask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _popcount_mask(bitmask_row: torch.Tensor, vocab_size: int) -> int:
        valid = 0
        for word in bitmask_row:
            bits = word.item()
            if bits < 0:
                bits += 1 << 32
            valid += bin(bits).count("1")
        return valid

    def _masked_argmax(
        self, logits: torch.Tensor, bitmask: torch.Tensor, device: str
    ) -> tuple[int, float]:
        """Grammar-masked argmax + density. Dispatches to Triton or CUDA fallback.

        Args:
            logits: 1-D float tensor [vocab_size], on GPU.
            bitmask: 1-D int32 tensor [num_words], on GPU (xgrammar bitmask).
            device: target device string.

        Returns:
            (token_id, density) where density = valid_tokens / vocab_size.
        """
        if self._backend == "triton" and logits.is_cuda:
            from velospec.triton.fused_logit_processor import fused_masked_argmax
            return fused_masked_argmax(logits, bitmask.to(device))

        # Fallback: xgrammar apply + PyTorch argmax + Python popcount
        xgr.apply_token_bitmask_inplace(
            logits.unsqueeze(0), bitmask.to(device)
        )
        masked = logits.unsqueeze(0)[0]
        valid = self._popcount_mask(bitmask[0], self._vocab_size)
        density = valid / self._vocab_size
        return masked.argmax().item(), density

    def _draft_k_tokens(
        self,
        input_ids: torch.Tensor,
        K: int,
        matcher: xgr.GrammarMatcher | None = None,
        bitmask: torch.Tensor | None = None,
    ) -> list[int]:
        """Autoregressively generate up to K tokens with the draft model (greedy)."""
        draft_device = getattr(self._draft, "device", "cpu")
        drafted: list[int] = []
        past_key_values = None
        current = input_ids.to(draft_device)

        with torch.inference_mode():
            for _ in range(K):
                outputs = self._draft(current, past_key_values=past_key_values, use_cache=True)
                next_logits = outputs.logits[:, -1, :].clone()

                if matcher is not None and bitmask is not None:
                    need_apply = matcher.fill_next_token_bitmask(bitmask)
                    if need_apply:
                        next_id, _ = self._masked_argmax(
                            next_logits[0], bitmask, draft_device
                        )
                    else:
                        next_id = next_logits.argmax(dim=-1).item()
                else:
                    next_id = next_logits.argmax(dim=-1).item()

                if next_id == self._eos_token_id:
                    break

                drafted.append(next_id)
                current = torch.tensor([[next_id]], device=draft_device)
                past_key_values = outputs.past_key_values

                if matcher is not None:
                    matcher.accept_token(next_id)
                    if matcher.is_terminated():
                        break

        return drafted

    # ------------------------------------------------------------------
    # Core: generate
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        schema: dict | list | str | None = None,
        max_tokens: int = 256,
    ) -> GenerationResult:
        """Generate text with grammar-constrained speculative decoding.

        Args:
            prompt: Input prompt string.
            schema: JSON schema (dict, JSON string, or file path).
                   Required for C2/C3/C4. Ignored for C1.
            max_tokens: Maximum tokens to generate.

        Returns:
            GenerationResult with text, acceptance rate, timing, and traces.
        """
        self._ensure_loaded()

        schema_str = self._normalize_schema(schema)

        if self.config in ("C2", "C3", "C4"):
            assert schema_str is not None, f"{self.config} requires a schema"

        prompt_tokens = self._tokenizer.encode(prompt, add_special_tokens=True)
        target_device = self.device

        # Build grammar pipeline
        matcher: xgr.GrammarMatcher | None = None
        bitmask: torch.Tensor | None = None
        if self.config in ("C2", "C3", "C4"):
            matcher, bitmask = self._build_grammar_pipeline(schema_str)

        generated: list[int] = []
        total_accepted = 0
        total_drafted = 0
        density_trace: list[float] = []
        k_trace: list[tuple[float, int]] = []
        rounds = 0
        K = self.K
        start = time.perf_counter()

        with torch.inference_mode():
            while len(generated) < max_tokens:
                prompt_len = len(prompt_tokens)
                prefix_len = prompt_len + len(generated)

                # ---- 1. Draft phase ----
                prefix = torch.tensor(
                    [prompt_tokens + generated], dtype=torch.long, device=target_device
                )

                current_K = K
                if self.config == "C4" and matcher is not None and bitmask is not None:
                    need_apply = matcher.fill_next_token_bitmask(bitmask)
                    if need_apply:
                        density = compute_density(bitmask[0], self._vocab_size)
                        current_K = adaptive_K(density)
                    else:
                        density = 1.0
                        current_K = K
                    k_trace.append((density, current_K))

                if self.config in ("C3", "C4") and matcher is not None and bitmask is not None:
                    draft_tokens = self._draft_k_tokens(
                        prefix, current_K, matcher=matcher, bitmask=bitmask,
                    )
                    if len(draft_tokens) > 0:
                        matcher.rollback(len(draft_tokens))
                else:
                    draft_tokens = self._draft_k_tokens(prefix, current_K)

                if not draft_tokens:
                    break

                n_draft = len(draft_tokens)

                # ---- 2. Verify phase ----
                verify_input = torch.tensor(
                    [prompt_tokens + generated + draft_tokens],
                    dtype=torch.long,
                    device=target_device,
                )
                verify_outputs = self._target(verify_input)
                verify_logits = verify_outputs.logits[0]

                # ---- 3. Accept / Reject (greedy) ----
                accepted_this_round = 0
                rejected = False
                verified_this_round = 0

                for i in range(n_draft):
                    pos = prefix_len - 1 + i
                    if pos >= verify_logits.shape[0]:
                        break
                    verified_this_round += 1
                    position_logits = verify_logits[pos].clone()

                    if self.config in ("C2", "C3", "C4") and matcher is not None and bitmask is not None:
                        need_apply = matcher.fill_next_token_bitmask(bitmask)
                        if need_apply:
                            target_predicted, density = self._masked_argmax(
                                position_logits, bitmask, target_device
                            )
                            density_trace.append(density)
                        else:
                            target_predicted = position_logits.argmax().item()
                            density_trace.append(1.0)
                    else:
                        target_predicted = position_logits.argmax().item()

                    if target_predicted == draft_tokens[i]:
                        generated.append(draft_tokens[i])
                        accepted_this_round += 1
                        total_accepted += 1

                        if self.config in ("C2", "C3", "C4") and matcher is not None:
                            matcher.accept_token(draft_tokens[i])
                            if matcher.is_terminated():
                                break
                    else:
                        generated.append(target_predicted)
                        if self.config in ("C2", "C3", "C4") and matcher is not None:
                            matcher.accept_token(target_predicted)
                        rejected = True
                        break

                total_drafted += verified_this_round

                # ---- 4. Bonus token ----
                if accepted_this_round == n_draft and not rejected:
                    bonus_pos = prefix_len + n_draft - 1
                    if bonus_pos < verify_logits.shape[0]:
                        bonus_logits = verify_logits[bonus_pos].clone()

                        if self.config in ("C2", "C3", "C4") and matcher is not None and bitmask is not None:
                            need_apply = matcher.fill_next_token_bitmask(bitmask)
                            if need_apply:
                                bonus_token, density = self._masked_argmax(
                                    bonus_logits, bitmask, target_device
                                )
                                density_trace.append(density)
                            else:
                                bonus_token = bonus_logits.argmax().item()
                                density_trace.append(1.0)
                        else:
                            bonus_token = bonus_logits.argmax().item()

                        if bonus_token == self._eos_token_id:
                            break

                        generated.append(bonus_token)
                        if self.config in ("C2", "C3", "C4") and matcher is not None:
                            matcher.accept_token(bonus_token)
                            if matcher.is_terminated():
                                break

                rounds += 1

                if self._eos_token_id in generated:
                    eos_idx = generated.index(self._eos_token_id)
                    generated = generated[:eos_idx]
                    break

        elapsed = time.perf_counter() - start
        acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        tps = len(generated) / max(elapsed, 1e-9)

        return GenerationResult(
            text=text,
            token_ids=generated,
            accepted_count=total_accepted,
            drafted_count=total_drafted,
            acceptance_rate=acceptance_rate,
            time_sec=elapsed,
            tokens_per_sec=tps,
            rounds=rounds,
            config=self.config,
            density_trace=density_trace,
            k_trace=k_trace,
        )
