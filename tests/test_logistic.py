"""Small pytest checks for the NumPy binary logistic regression."""

import numpy as np
from src.linear_models import LogisticRegression


def test_logistic_regression_basic():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(n_iterations=500, lr=0.1)
    model.fit(X, y)

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (4,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.array_equal(model.predict(X), y)
    assert model.losses_history[-1] < model.losses_history[0]


def test_higher_threshold_does_not_increase_positive_predictions():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(n_iterations=500, lr=0.1)
    model.fit(X, y)

    assert np.all(model.predict(X, threshold=0.9) <= model.predict(X, threshold=0.5))
