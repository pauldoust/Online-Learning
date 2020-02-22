"""
Online Passive-Agressive Algorithms
"""
import numpy as np

class PA:
    def __init__(self, C = 0.5, update_ = 'fr'):
        """
        Inputs C, update strategy update_
        
        """
        self.C = C # Slack
        self.n = None
        self.update_ = update_
        
    def _one_fit(self, x, y):
        """
        x: One instance
        y: True labels (+1,-1)
        """
        l = x.shape[0] # No. of features
        
        # Predict y
        y_hat = np.sign(np.dot(self.w,x))

        # Compute loss
        second_term = (np.sum(np.dot(self.w,x)))*y
        #print(second_term)
        loss = max([0, 1-second_term])

        tau = self._find_tau(loss, x, self.C) # Find tau

        # updated weight
        self.w = self.w + tau*y*x
        return self.w, loss

    def _find_tau(self, loss, x, C):
        
        x_abs = np.sum(np.dot(x,x))
        st = loss/x_abs
        
        if self.update_ == "classic":
            temp = st
        if self.update_ == "fr":
            temp = min([C, st])
        if self.update_ == "sr":
            temp = loss/(x_abs + (2*self.C)**(-1))
        #tau = min([C, st])
        tau = temp
        return tau

    def fit(self, X, Y):
        self.n = X.shape[0]
        self.w = np.zeros(np.shape(X[1]))
        for t in range(self.n):
            self.w, loss = self._one_fit(X[t,:], Y[t])

        return self.w, loss

    def predict(self, x):
        y_pred = np.sign(np.dot(self.w, x))

        return y_pred


    def predict_all(self, x):
        test_size = x.shape[0]
        y_pred = np.zeros(test_size, dtype = 'int')
        for i in range(test_size):
            y_pred[i] = self.predict(x[i])

        return y_pred
