import numpy as np
from . import losses, optimizers     
from scipy.special import expit 

class LinearRegression():
    def __init__(self, n_iterations=100, lr=0.01, loss = 'mse'):

        self.n_iterations = n_iterations
        self.lr = lr
        self.loss = loss

        self.weights = None
        self.bias = None

        available_losses = ['mse', 'mae']
        if self.loss not in available_losses:
            raise ValueError(f"Unknown loss function: '{self.loss}'. "f"Available losses: {', '.join(available_losses)}")

    def fit(self, X, y):
        y_ = y.copy()
        self.losses_history = []
        self.weights = np.zeros(X.shape[1])
        self.bias = 1

        for i in range (self.n_iterations):

            y_pred = X@self.weights + self.bias
            if self.loss == 'mse':
                loss = losses.mse(y_, y_pred)
                self.losses_history.append(loss.round(3))
                dl_dypred = losses.mse_grad(y_, y_pred)
            else:
                loss = losses.mae(y_, y_pred)
                self.losses_history.append(loss.round(3))
                dl_dypred = losses.mae_grad(y_, y_pred)
            grads = self.parametrs_gradients(X, dl_dypred)
            self.weights = self.weights - self.lr*grads[0]
            self.bias = self.bias - self.lr*grads[1] 


    def parametrs_gradients(self, X, dl_dypred):
        grad_w = X.T @ dl_dypred
        grad_b = np.sum(dl_dypred)
        return grad_w, grad_b
    
    def predict(self, X):
        return X@self.weights + self.bias
    
class LogisticRegression():
    def __init__(self, n_iterations=100, lr=0.1, loss='bce'):
        self.n_iterations = n_iterations
        self.lr = lr
        self.loss = loss

        self.weights = None
        self.bias = None

    def fit(self, X, y):

        y_true = y.copy()
        self.weights = np.zeros(X.shape[1])
        self.bias = 1
        self.losses_history = []

        for _ in range(self.n_iterations):
            z_pred = X@self.weights + self.bias
            probs = self.sigmoid(z_pred)

            loss = losses.BCE(y_true, probs)
            self.losses_history.append(loss)
            dl_dz = losses.BCE_grad(y_true, probs)
            grad_w, grad_b = self.parametr_gradients(X, dl_dz)
            self.weights = self.weights - self.lr*grad_w
            self.bias = self.bias - self.lr*grad_b

    def predict_proba(self, X):
        return self.sigmoid(X@self.weights + self.bias)
    
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def parametr_gradients(self, X, dl_dz):
        grad_w = X.T @ dl_dz
        grad_b = np.sum(dl_dz)
        return grad_w, grad_b



                
            

            

            



