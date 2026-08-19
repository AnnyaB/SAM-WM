import numpy as np
import pytest

from coolworld.sam import Ellipsoid, exact_linear_support_intersection, support_deficiency


def test_support_deficiency_spectrum_and_symmetry() -> None:
    sigma = np.diag([0.0, 1.0, 9.0]).astype(np.float64)
    d = support_deficiency(sigma, tau=1.0)
    assert np.allclose(d, d.T)
    eig = np.linalg.eigvalsh(d)
    assert np.all(eig > 0)
    assert np.all(eig <= 1.0 + 1e-12)
    assert np.allclose(np.diag(d), [1.0, 0.5, 0.1])


def test_support_deficiency_orthogonal_equivariance() -> None:
    sigma = np.array([[2.0, 0.3], [0.3, 0.7]], dtype=np.float64)
    q = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    lhs = support_deficiency(q @ sigma @ q.T, tau=0.2)
    rhs = q @ support_deficiency(sigma, tau=0.2) @ q.T
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_exact_intersection_support_returns_real_interval() -> None:
    pytest.importorskip("cvxpy")
    # Deterministic mathematical verification values; this is not an empirical dataset.
    data = Ellipsoid(
        center=np.array([0.0, 0.0]),
        precision=np.eye(2),
        radius=1.0,
    )
    mechanism = Ellipsoid(
        center=np.array([0.5, 0.0]),
        precision=np.eye(2),
        radius=1.0,
    )
    interval = exact_linear_support_intersection(np.array([1.0, 0.0]), data, mechanism)
    assert interval.status == "IDENTIFIED_SET"
    assert interval.lower <= interval.upper
    assert abs(interval.lower + 0.5) < 2e-4
    assert abs(interval.upper - 1.0) < 2e-4


def test_exact_intersection_detects_conflict() -> None:
    pytest.importorskip("cvxpy")
    data = Ellipsoid(np.array([0.0]), np.eye(1), 0.2)
    mechanism = Ellipsoid(np.array([2.0]), np.eye(1), 0.2)
    interval = exact_linear_support_intersection(np.array([1.0]), data, mechanism)
    assert interval.status == "CONFLICT"
