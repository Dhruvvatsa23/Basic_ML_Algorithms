#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class LogisticRegression:
    def __init__(self, lr = 0.001, n_iters = 100):
        self.lr = lr
        self.n_iter = n_inters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):  #keeping the sickit_learn library 
                          #in mind these arguments were passed
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        #gradient descent iteratively updating weights and bias 
        
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            #derivative part 
            dw = (1/n_samples)*np.dot((X.T, y_predicted - y))
            db = (1/n_samples)*np.sum( y_predicted - y)
            
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
            
    def predict(self, X): #test samples we want to predict
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = (1 if i > 0.5 else 0 for i in y_predicted)
        return y_predicted_cls
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# In[ ]:




