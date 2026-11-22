import numpy as np
import pytest
from app.routing.core import FeatureRouter, lagrange_predict, validate_features

FEATURES = {
    "role": "faculty",
    "discipline": ["engineering", "medical"],
    "study_level": "bachelor",
    "experience": 0.2,
    "task_type": "quiz",
}


def test_feature_vector_rejects_group_labels_as_model_inputs():
    a = validate_features(FEATURES)
    # Group-label permutations cannot enter the feature model: each is rejected as an unknown feature.
    from itertools import permutations

    labels = [0, 1, 2, 3]
    for perm in permutations(labels):
        with pytest.raises(ValueError, match="Unexpected model feature keys"):
            validate_features({**FEATURES, "group_labels": perm})
    assert np.array_equal(a, validate_features(FEATURES))


def test_experience_validation():
    bad = {**FEATURES, "experience": 1.1}
    with pytest.raises(ValueError):
        validate_features(bad)


def test_cold_start_without_measured_threshold():
    r = FeatureRouter(min_ratings=None).recommend(FEATURES, [], ["a", "b"], {"quiz": "b"})
    assert r["mode"] == "cold_start" and r["template_id"] == "b"


def test_lagrange_prediction_at_nodes():
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([1.0, 0.0, 1.0])
    assert np.allclose(lagrange_predict(x, y, x), y)


def test_group_baseline_methods():
    from app.routing.core import predict_group_baseline

    labels = np.array([0, 1, 2, 3])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert predict_group_baseline("lagrange_equispaced", labels, values, 2) == pytest.approx(3.0)
    assert np.isfinite(predict_group_baseline("lagrange_chebyshev", labels, values, 2))
    assert predict_group_baseline("cubic_spline", labels, values, 2) == pytest.approx(3.0)
