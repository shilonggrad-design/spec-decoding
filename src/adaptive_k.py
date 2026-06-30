"""
adaptive_k.py — Density-driven adaptive K controller.

Low density (few valid tokens) → draft accuracy high → speculate aggressively (large K)
High density (many valid tokens) → draft accuracy low → speculate conservatively (small K)

Based on SpecDec++ (ICML 2024) threshold policy.
Grammar density replaces trained acceptance head — zero-cost, deterministic, forward-looking.
"""

from __future__ import annotations

# Qwen3.5 vocab_size = 248,320
# density_threshold = 0.005 → 0.005 * 248320 ≈ 1241 valid tokens
# density_threshold * 4 = 0.02 → 0.02 * 248320 ≈ 4966 valid tokens

K_MIN = 1
K_MAX = 8
DENSITY_THRESHOLD = 0.005


def compute_density(bitmask_row, vocab_size: int) -> float:
    """Count valid tokens in a packed int32 bitmask row / vocab_size.

    Args:
        bitmask_row: 1D tensor of int32, length = ceil(vocab_size / 32)
        vocab_size: total vocabulary size (e.g. 248320)

    Returns:
        density in [0, 1] — fraction of valid tokens
    """
    valid = 0
    for word in bitmask_row:
        bits = word.item()
        if bits < 0:
            bits += 1 << 32
        valid += bin(bits).count("1")
    return valid / vocab_size


def adaptive_K(
    density: float,
    K_min: int = K_MIN,
    K_max: int = K_MAX,
    density_threshold: float = DENSITY_THRESHOLD,
) -> int:
    """Map grammar mask density to speculation width K.

    Thresholds tuned for Qwen3.5 (vocab=248,320):
    - density < 0.005  (< ~1241 valid tokens)  → K=K_max (speculate aggressively)
    - density < 0.02   (< ~4966 valid tokens)  → K=(K_min+K_max)//2 (moderate)
    - else             (loose constraint)       → K=K_min (speculate conservatively)

    Examples (from Week 1 density trace):
        JSON key boundary:   density=0.00001 → K=8  (draft can't miss)
        String value:        density=0.98    → K=1  (draft likely diverges)
        Numeric field:       density=0.003   → K=8
        Enum field:          density=0.00006 → K=8

    Returns:
        K (int) — number of tokens to draft this round
    """
    if density < density_threshold:
        return K_max
    elif density < density_threshold * 4:
        return (K_min + K_max) // 2
    else:
        return K_min
