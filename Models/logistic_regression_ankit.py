import numpy as np

class LogisticRegressionBinary:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.iterations = n_iters
        self.theta_ = None  # parameters

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def calculate_gradient(self, theta, X, y):
        m = y.size  # number of training examples
        return (X.T @ (self.sigmoid(X @ theta) - y)) / m

    def fit(self, X, y, tol=1e-6):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        self.theta_ = np.zeros(X_b.shape[1])  # initialize parameters

        for i in range(self.iterations):
            grad = self.calculate_gradient(self.theta_, X_b, y)
            self.theta_ -= self.learning_rate * grad

            if np.linalg.norm(grad) < tol:
                print(f"Converged at iteration {i}")
                break

        return self  # Return self for method chaining

    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        return self.sigmoid(X_b @ self.theta_)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    

class LogisticRegressionMultiClass(LogisticRegressionBinary):

    def __init__(self, learning_rate=0.001, n_iters=1000):
        super().__init__(learning_rate, n_iters)
        self.classes_ = None
        self.thetas_ = None

    def fit(self, X, y, tol=1e-6):
        self.classes_ = np.unique(y)
        self.thetas_ = []

        for cls in self.classes_:
            y_binary = (y == cls).astype(int)
            # Create a new binary classifier for each class
            binary_clf = LogisticRegressionBinary(self.learning_rate, self.iterations)
            binary_clf.fit(X, y_binary, tol)
            self.thetas_.append(binary_clf.theta_)

        self.thetas_ = np.array(self.thetas_)
        return self  # Return self for method chaining

    def predict_proba(self, X):
        """Return probability for each class"""
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        probabilities = np.array([self.sigmoid(X_b @ theta) for theta in self.thetas_]).T
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]