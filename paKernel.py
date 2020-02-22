"""
Kernelized online Passive-Agressive Algorithms

"""
import numpy as np
from pa import PA
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

class KPA:
    def __init__(self, C = 0.5, update_ = 'fr', kernel = 'rbf', gamma = 0.5, degree = 3):
        """
        C: For slacks [Trade-off]
        update_ : 'classic' or 'fr'
        kernel: 'rbf' (default), 'poly'
        """
        self.C = C 
        self.n = None
        self.alpha = None
        self.update_ = update_
        self.kernel_ = kernel
        self.gamma_ = gamma
        self.degree_ = degree
        self.support_vectors = None
        
    def _one_fit(self, x, y):
        """
        x: One instance
        y: True label of the instance (+1 or -1)
        """

        x = np.array(x)
        x = np.append(x,np.ones((1,1))).reshape(1,-1)

        if self.support_vectors is None:
            lc = 0
        else:
            if self.kernel_ == 'rbf':
                #print(x,self.support_vectors)
                kernel = rbf_kernel(self.support_vectors, x, gamma=self.gamma_)
            elif self.kernel_ == 'poly':
                kernel = polynomial_kernel(self.support_vectors, x, degree=self.degree_)
            lc = np.sum(np.transpose(kernel)*self.alpha)

        l = x.shape[0] # No. of features
        loss = max(0,1-y*lc)
        
        return x,loss

    def fit(self, X, Y):
        self.n = X.shape[0]
        for t in range(self.n):
            x = X[t,:]
            y = Y[t]
            i,loss = self._one_fit(X[t,:], Y[t])
            if not loss == 0:
                #print(loss)
                if self.update_ == "classic":
                    if self.kernel_ == 'linear':
                        print('For Linear Kernel - class PA in pa.py')
                    elif self.kernel_ == 'rbf':
                        new_alpha = y*loss / (rbf_kernel(i,i, gamma=self.gamma_)[0][0])
                    elif self.kernel_ == 'poly':
                        new_alpha = y*loss / (polynomial_kernel(i,i, degree=self.degree_)[0][0])

                else:
                    if self.kernel_ == 'linear':
                        z = 'For Linear Kernel - class PA in pa.py'
                    elif self.kernel_ == 'rbf':
                        new_alpha = y*min([self.C, loss / (rbf_kernel(i,i,gamma=self.gamma_)[0][0])])
                    elif self.kernel_ == 'poly':
                        new_alpha = y*min([self.C, loss / (polynomial_kernel(i,i, degree=self.degree_)[0][0])])
                if self.kernel_ != 'linear':
                    if self.alpha is None:
                        self.alpha = np.array([new_alpha])
                        self.support_vectors = i
                    else:
                        self.alpha = np.hstack([self.alpha,new_alpha])
                        self.support_vectors = np.vstack([self.support_vectors,i])
                    z = 'Training complete.'
    
        return z

    def predict(self, x):
        instance = np.array(x)
        instance = np.append(instance,np.ones((1,1))).reshape(1,-1)
        if self.kernel_ == 'rbf':
            kernel = rbf_kernel(self.support_vectors,instance,gamma=self.gamma_)
        elif self.kernel_ =='linear':
            print('For Linear Kernel - check class PA in pa.py')
        elif self.kernel_ == 'poly':
            kernel = polynomial_kernel(self.support_vectors,instance,degree=self.degree_)
        y_pred = np.sign(np.sum(np.transpose(kernel)*self.alpha))
        #y_pred = np.sum(self.find_kernel(self.support_vectors, x))
        
        return y_pred #kernel, self.alpha#, self.support_vectors
    
    def predict_all(self,X):
        test_size= X.shape[0]
        y_pred= np.zeros(test_size, dtype='int')
        for i in range(test_size):
            y_pred[i]= self.predict(X[i,:])
        return y_pred