"""Linear solves, fitting, decompositions, and a tiny PCA workflow."""

import numpy as np


TOLERANCES = {
    "solve": (1e-12, 1e-12),
    "fit_coefficients": (1e-12, 1e-12),
    "eigenvalues": (1e-12, 1e-12),
    "principal_axis": (1e-12, 1e-12),
    "singular_values": (1e-12, 1e-12),
    "qr_diagonal": (1e-12, 1e-12),
    "reconstruction": (1e-12, 1e-12),
}


def _positive_anchor(vector):
    anchor = np.argmax(np.abs(vector))
    return vector * np.where(vector[anchor] < 0.0, -1.0, 1.0)


def results():
    expected = np.array([1.0, -2.0, 3.0])
    system = np.array([[4.0, 1.0, 2.0], [0.0, 3.0, -1.0], [2.0, -2.0, 5.0]])
    rhs = system @ expected
    solved = np.linalg.solve(system, rhs)

    x = np.linspace(-2.0, 2.0, 9)
    design = np.column_stack((np.ones_like(x), x, x * x))
    observations = 2.0 - 3.0 * x + 0.5 * x * x
    fitted, _, _, _ = np.linalg.lstsq(design, observations, rcond=None)

    samples = np.array([
        [2.0, 1.0, 0.0], [3.0, 2.0, 1.0], [4.0, 1.0, 2.0],
        [5.0, 3.0, 1.0], [6.0, 4.0, 3.0], [7.0, 3.0, 4.0],
    ])
    centered = samples - samples.mean(axis=0)
    covariance = centered.T @ centered / (samples.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order].real
    principal_axis = _positive_anchor(eigenvectors[:, order[0]].real)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    q, r = np.linalg.qr(design)
    reconstruction = q @ r
    return {
        "solve": solved,
        "fit_coefficients": fitted,
        "eigenvalues": eigenvalues,
        "principal_axis": principal_axis,
        "singular_values": singular_values,
        "qr_diagonal": np.diag(r),
        "reconstruction": reconstruction,
    }


def main():
    out = results()
    print("solved coefficients:", np.round(out["solve"], 6))
    print("quadratic fit:", np.round(out["fit_coefficients"], 6))
    print("PCA eigenvalues:", np.round(out["eigenvalues"], 6))
    print("first principal axis:", np.round(out["principal_axis"], 6))
    assert np.allclose(out["solve"], [1.0, -2.0, 3.0], rtol=0.0, atol=1e-12)
    assert np.allclose(out["fit_coefficients"], [2.0, -3.0, 0.5], rtol=0.0, atol=1e-12)
    assert np.allclose(out["reconstruction"][:, 0], np.ones(9), rtol=0.0, atol=1e-12)
    assert np.allclose(out["singular_values"], [5.61825335, 1.86292763, 0.79460468], rtol=0.0, atol=1e-8)


if __name__ == "__main__":
    main()
