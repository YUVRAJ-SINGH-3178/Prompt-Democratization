from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from scipy.interpolate import BarycentricInterpolator, CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


@dataclass(frozen=True)
class Rating:
    features: tuple[float, ...]
    template_id: str
    rating: float
    edited: bool = False
    timestamp: str = ""
    synthetic: bool = False


class Prediction(TypedDict):
    template_id: str
    mean: float
    std: float
    n: int
    synthetic: bool


def validate_features(features: dict[str, object]) -> np.ndarray:
    required = ("role", "discipline", "study_level", "experience", "task_type")
    if any(k not in features for k in required):
        raise ValueError("Feature vector requires role, discipline, study_level, experience and task_type")
    unknown = set(features) - set(required)
    if unknown:
        raise ValueError(f"Unexpected model feature keys: {', '.join(sorted(unknown))}")

    def onehot(name: str, val: object, multi: bool = False) -> list[float]:
        if multi:
            if not isinstance(val, (list, tuple)):
                raise ValueError(f"{name} must be a list of category names")
            vals = list(val)
        else:
            vals = [val]
        if not vals or any(not isinstance(x, str) or not x for x in vals):
            raise ValueError(f"{name} must contain category names")
        # Fixed category dictionary is defined from explicit feature vocabularies, never group ids.
        vocab = {
            "role": ["faculty", "ta", "student", "staff"],
            "discipline": ["engineering", "medical", "arts", "science", "business", "other"],
            "study_level": ["bachelor", "master", "doctoral", "other"],
            "task_type": ["quiz", "worksheet", "lesson", "assessment", "other"],
        }[name]
        unknown = set(vals) - set(vocab)
        if unknown:
            raise ValueError(f"Unknown {name} categories: {', '.join(sorted(unknown))}")
        return [float(x in vals) for x in vocab]

    exp = features["experience"]
    if isinstance(exp, bool) or not isinstance(exp, (int, float)) or not -1 <= exp <= 1:
        raise ValueError("experience must be numeric in [-1, 1]")
    v = (
        onehot("role", features["role"])
        + onehot("discipline", features["discipline"], True)
        + onehot("study_level", features["study_level"])
        + [float(exp)]
        + onehot("task_type", features["task_type"])
    )
    return np.asarray(v, dtype=float)


class FeatureRouter:
    def __init__(self, confidence_threshold: float = 0.25, min_ratings: int | None = None, random_state: int = 42):
        if confidence_threshold <= 0 or (min_ratings is not None and min_ratings < 1):
            raise ValueError("Threshold and minimum ratings must be positive")
        self.confidence_threshold, self.min_ratings, self.random_state = confidence_threshold, min_ratings, random_state

    def recommend(
        self,
        features: dict[str, object],
        ratings: list[Rating],
        template_ids: list[str],
        defaults: dict[str, str],
        shortlist_size: int = 3,
    ) -> dict:
        x = validate_features(features)
        if not template_ids:
            raise ValueError("At least one template is required")
        eligible = [
            t
            for t in template_ids
            if self.min_ratings is not None
            and sum(r.template_id == t and not r.synthetic for r in ratings) >= self.min_ratings
        ]
        if not eligible:
            task = str(features["task_type"])
            chosen = defaults.get(task)
            if chosen not in template_ids:
                return {
                    "mode": "cold_start_unconfigured",
                    "template_id": None,
                    "shortlist": template_ids[:],
                    "predictions": [],
                    "message": f"Set an approved default template for task type {task}.",
                    "synthetic": False,
                }
            return {
                "mode": "cold_start",
                "template_id": chosen,
                "shortlist": template_ids[:],
                "predictions": [],
                "synthetic": False,
            }
        pred: list[Prediction] = []
        for tid in eligible:
            rows = [r for r in ratings if r.template_id == tid and not r.synthetic]
            X = np.asarray([r.features for r in rows], dtype=float)
            y = np.asarray([r.rating for r in rows], dtype=float)
            if X.ndim != 2 or X.shape[1] != len(x):
                continue
            kernel = ConstantKernel(0.1, (1e-3, 10)) * RBF(0.5, (0.05, 5)) + WhiteKernel(1e-3, (1e-8, 1))
            model = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=self.random_state
            )
            model.fit(X, y)
            mean, std = model.predict(x.reshape(1, -1), return_std=True)
            pred.append(
                {"template_id": tid, "mean": float(mean[0]), "std": float(std[0]), "n": len(rows), "synthetic": False}
            )
        pred.sort(key=lambda z: (-z["mean"], z["template_id"]))
        if not pred:
            chosen = defaults.get(str(features["task_type"]))
            return {
                "mode": "cold_start" if chosen else "cold_start_unconfigured",
                "template_id": chosen,
                "shortlist": template_ids,
                "predictions": [],
                "synthetic": False,
            }
        sure = pred[0]["std"] < self.confidence_threshold
        return {
            "mode": "selected" if sure else "shortlist",
            "template_id": pred[0]["template_id"] if sure else None,
            "shortlist": [p["template_id"] for p in pred[: max(1, shortlist_size)]],
            "predictions": pred,
            "synthetic": False,
        }


def lagrange_predict(x: np.ndarray, y: np.ndarray, query: np.ndarray, nodes: str = "equispaced") -> np.ndarray:
    n = len(x)
    if nodes == "equispaced":
        domain = np.linspace(-1, 1, n)
    elif nodes == "chebyshev":
        k = np.arange(n)
        domain = np.cos((2 * k + 1) * np.pi / (2 * n))
        order = np.argsort(domain)
        domain, y = domain[order], y[order]
    else:
        raise ValueError("nodes must be equispaced or chebyshev")
    return np.asarray(BarycentricInterpolator(domain, y)(query), dtype=float)


def predict_baseline(method: str, x: np.ndarray, y: np.ndarray, query: np.ndarray, seed: int = 42) -> np.ndarray:
    if method in ("lagrange_equispaced", "lagrange_chebyshev"):
        return lagrange_predict(x, y, query, "equispaced" if method.endswith("equispaced") else "chebyshev")
    if method == "cubic_spline":
        order = np.argsort(x)
        return np.asarray(CubicSpline(x[order], y[order], bc_type="natural", extrapolate=True)(query))
    raise ValueError(f"Unsupported interpolation baseline {method}")


def predict_group_baseline(method: str, group_labels: np.ndarray, values: np.ndarray, query_label: int) -> float:
    """Interpolate real per-group ratings. Integer labels are isolated to this legacy baseline."""
    from scipy.interpolate import BarycentricInterpolator

    labels = np.asarray(group_labels, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(labels) < 2:
        raise ValueError("At least two real group ratings are required for interpolation")
    order = np.argsort(labels)
    labels, y = labels[order], y[order]
    if method == "lagrange_equispaced":
        return float(BarycentricInterpolator(labels, y)(float(query_label)))
    if method == "lagrange_chebyshev":
        n = len(labels)
        nodes = np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
        idx = np.argsort(nodes)
        return float(BarycentricInterpolator(nodes[idx], y)(2 * query_label / 3 - 1))
    if method == "cubic_spline":
        if len(labels) < 3:
            raise ValueError("Cubic spline requires at least three distinct real group ratings")
        return float(CubicSpline(labels, y, bc_type="natural", extrapolate=True)(float(query_label)))
    raise ValueError(f"Unsupported group baseline {method}")
