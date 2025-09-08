import numpy as np

class LogisticRegression():
    def __init__(self, lr, n_iter):
        self.lr = lr # Learning rate of the model
        self.n_iter = n_iter # Iterations of Gradient Descent
        
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def fit(self, X, y, rand_seed=None):
        self.X = X
        self.y = y
        self.height, self.width = X.shape

        if rand_seed is not None:
            np.random.seed(rand_seed) # Manual seed for replication purposes
            self.weights = np.random.rand(self.width) # Random weights for starting purposes
        else:
            self.weights = np.zeros(self.width) # else make it zeroes
        
        self.bias = 0 
        
        for _ in range(self.n_iter):
            self.grad_descent() # Gradient descent 
        
        return self
            
    def predict_probs(self, X):
        return self.sigmoid(X.dot(self.weights) + self.bias)
    
    def predict(self, X, threshold=0.5):
        probs = self.predict_probs(X)
        return (probs >= threshold).astype(int) # Create an array of True and False and convert it to integers (either 0 or 1) for classification
        
    def grad_descent(self):
        y_pred = self.predict_probs(self.X) # Running sigmoid on predictions for classification
        
        weights_gd = (self.X.T).dot(y_pred - self.y) / self.height # Gradient descent for weights
        bias_gd = (y_pred - self.y).mean() # Gradient descent for bias
        
        self.weights -= (self.lr * weights_gd) # Adjust weights
        self.bias -= (self.lr * bias_gd) # Adjust bias