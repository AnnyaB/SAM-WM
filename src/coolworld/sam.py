from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def support_deficiency(covariance: FloatArray, tau: float) -> FloatArray:
    """Compute D_tau(Sigma) = tau (Sigma + tau I)^(-1).

    The solve form avoids explicitly forming a matrix inverse. Input must be a
    finite symmetric PSD covariance matrix and tau must be strictly positive.
    """

    sigma = np.asarray(covariance, dtype=np.float64)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("covariance contains non-finite values")
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("tau must be positive and finite")
    if not np.allclose(sigma, sigma.T, atol=1e-10, rtol=1e-10):
        raise ValueError("covariance must be symmetric")
    if np.linalg.eigvalsh(sigma).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")

    eye = np.eye(sigma.shape[0], dtype=np.float64)
    return tau * np.linalg.solve(sigma + tau * eye, eye)


@dataclass(frozen=True, slots=True)
class Ellipsoid:
    center: FloatArray
    precision: FloatArray
    radius: float

    def validate(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        precision = np.asarray(self.precision, dtype=np.float64)
        if center.ndim != 1:
            raise ValueError("ellipsoid center must be a vector")
        if precision.shape != (center.size, center.size):
            raise ValueError("precision shape must match center dimension")
        if not np.allclose(precision, precision.T, atol=1e-10, rtol=1e-10):
            raise ValueError("precision must be symmetric")
        if np.linalg.eigvalsh(precision).min() <= 0:
            raise ValueError(
                "reference exact solver currently requires positive-definite precision"
            )
        if self.radius <= 0 or not np.isfinite(self.radius):
            raise ValueError("radius must be positive and finite")


@dataclass(frozen=True, slots=True)
class SupportInterval:
    lower: float
    upper: float
    status: str


def exact_linear_support_intersection(
    query: FloatArray,
    data_set: Ellipsoid,
    mechanism_set: Ellipsoid,
    *,
    solver: str = "CLARABEL",
) -> SupportInterval:
    """Exact convex reference projection for C_D ∩ C_M in the linear case.

    Solves two convex QCQPs: min/max q^T b subject to both ellipsoid constraints.
    This is a reference implementation for paper verification, not a claim that
    nonlinear neural-world-model identification is solved by the same theorem.
    """

    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError(
            "exact SAM reference solver requires cvxpy; install project dependencies"
        ) from exc

    data_set.validate()
    mechanism_set.validate()
    q = np.asarray(query, dtype=np.float64)
    d = np.asarray(data_set.center, dtype=np.float64).size
    if q.shape != (d,):
        raise ValueError("query dimension does not match ellipsoids")

    b = cp.Variable(d)
    constraints = [
        cp.quad_form(b - data_set.center, data_set.precision) <= data_set.radius**2,
        cp.quad_form(b - mechanism_set.center, mechanism_set.precision) <= mechanism_set.radius**2,
    ]

    values: list[float] = []
    for sign in (1.0, -1.0):
        problem = cp.Problem(cp.Minimize(sign * (q @ b)), constraints)
        problem.solve(solver=solver)
        if problem.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
            return SupportInterval(float("nan"), float("nan"), "CONFLICT")
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"unexpected convex solver status: {problem.status}")
        assert problem.value is not None
        values.append(float(problem.value))

    lower = values[0]
    upper = -values[1]
    return SupportInterval(lower=lower, upper=upper, status="IDENTIFIED_SET")
