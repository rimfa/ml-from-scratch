import numpy as np
from src.linear_models import LinearRegression

def test_linear_regression():
    X = np.array([[-2], [-1], [0], [1], [2]])
    y = 3 * X.ravel() + 12

    model = LinearRegression(n_iterations=1000, lr=0.01)
    model.fit(X, y)

    assert np.isclose(model.weights[0], 3)
    assert np.isclose(model.bias, 12)
