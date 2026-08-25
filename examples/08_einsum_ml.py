"""Attention-style einsum contractions, softmax, and batched matmul."""

import numpy as np


TOLERANCES = {
    "scores": (1e-12, 1e-12),
    "weights": (1e-12, 1e-12),
    "context": (1e-12, 1e-12),
    "batched_scores": (1e-12, 1e-12),
}


def results():
    rng = np.random.default_rng(314159)
    queries = rng.normal(size=(2, 3, 4))
    keys = rng.normal(size=(2, 5, 4))
    values = rng.normal(size=(2, 5, 3))
    scores = np.einsum("bqd,bkd->bqk", queries, keys) / np.sqrt(queries.shape[-1])
    shifted = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    weights /= weights.sum(axis=-1, keepdims=True)
    context = np.einsum("bqk,bkv->bqv", weights, values)
    batched_scores = queries @ keys.swapaxes(-1, -2) / 2.0
    return {
        "scores": scores,
        "weights": weights,
        "weight_sums": weights.sum(axis=-1),
        "context": context,
        "batched_scores": batched_scores,
    }


def main():
    out = results()
    print("attention row sums:", out["weight_sums"])
    print("first attention weights:", np.round(out["weights"][0, 0], 6))
    print("first context vector:", np.round(out["context"][0, 0], 6))
    assert np.allclose(out["weight_sums"], np.ones((2, 3)), rtol=0.0, atol=1e-15)
    assert np.allclose(out["scores"], out["batched_scores"], rtol=0.0, atol=1e-12)
    expected_weights = [0.31145789, 0.36088471, 0.12725665, 0.11079817, 0.08960259]
    assert np.allclose(out["weights"][0, 0], expected_weights, rtol=0.0, atol=1e-8)


if __name__ == "__main__":
    main()
