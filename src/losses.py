import numpy as np

def mse(y_true, y_pred):
    return (sum((y_pred-y_true)**2))/len(y_true)
def mse_grad(y_true, y_pred):
    errors = y_pred-y_true
    return 2/len(y_true) * (errors)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))
def mae_grad(y_true, y_pred):
    errors = y_pred - y_true
    return np.sign(errors) / len(y_true)

def BCE(y_true, probs):
    return -np.mean((y_true * np.log(probs)) + (1 - y_true) * np.log(1 - probs))
def BCE_grad(y_true, probs):
    error = probs-y_true
    return (error)/len(y_true)
