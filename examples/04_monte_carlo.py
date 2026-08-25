"""Seeded simulation and bootstrap confidence intervals with both RNG APIs."""

import numpy as np


TOLERANCES = {"bootstrap_ci": (1e-12, 1e-12)}


def results():
    rng = np.random.default_rng(8675309)
    points = rng.uniform(-1.0, 1.0, size=(20_000, 2))
    inside = np.count_nonzero(np.sum(points * points, axis=1) <= 1.0)
    pi_estimate = 4.0 * inside / points.shape[0]

    observations = rng.normal(loc=12.0, scale=2.5, size=80)
    bootstrap_indices = rng.integers(0, observations.size, size=(1_000, observations.size))
    bootstrap_means = observations[bootstrap_indices].mean(axis=1)
    confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])

    legacy = np.random.RandomState(1969)
    legacy_rolls = legacy.randint(1, 7, size=24)
    return {
        "inside_count": np.array(inside),
        "pi_estimate": np.array(pi_estimate),
        "bootstrap_ci": confidence_interval,
        "legacy_rolls": legacy_rolls,
    }


def main():
    out = results()
    print("inside circle / 20000:", out["inside_count"])
    print("pi estimate:", out["pi_estimate"])
    print("bootstrap 95% CI:", np.round(out["bootstrap_ci"], 6))
    print("legacy first rolls:", out["legacy_rolls"][:8])
    assert np.array_equal(out["inside_count"], 15831)
    assert np.allclose(out["pi_estimate"], 3.1662, rtol=0.0, atol=0.0)
    assert np.array_equal(out["legacy_rolls"][:8], [2, 4, 5, 1, 6, 2, 4, 5])


if __name__ == "__main__":
    main()
