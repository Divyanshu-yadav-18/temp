import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Initialize the Perceptron.
        
        Args:
            learning_rate (float): The step size for weight updates (between 0.0 and 1.0).
            n_iters (int): Number of passes over the training dataset (epochs).
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Args:
            X (array-like): Training data of shape (n_samples, n_features).
            y (array-like): Target values (binary: 0 or 1) of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Ensure labels are explicitly 1 or 0 for the step function
        y_ = np.array([1 if i > 0 else 0 for i in y])

        # Training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # 1. Calculate the linear combination (z = W·X + b)
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # 2. Apply the activation function (Step function)
                y_predicted = self._unit_step_func(linear_output)
                
                # 3. Calculate the update step
                # update = learning_rate * (Actual - Predicted)
                update = self.lr * (y_[idx] - y_predicted)
                
                # 4. Update weights and bias
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Predict the class labels for new input data.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._unit_step_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        """
        The Heaviside step function: returns 1 if x >= 0, else 0.
        """
        return np.where(x >= 0, 1, 0)
