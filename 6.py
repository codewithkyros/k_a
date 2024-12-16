import numpy as np

# Define the Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, max_iter=1000):
        self.weights = np.zeros(input_size)  # Initialize weights to zero
        self.bias = 0  # Initialize bias to zero
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def activation(self, x):
        # Step function as the activation
        return 1 if x >= 0 else 0

    def train(self, X, y):
        for iteration in range(self.max_iter):
            error_count = 0
            for xi, target in zip(X, y):
                # Calculate weighted sum
                weighted_sum = np.dot(xi, self.weights) + self.bias
                prediction = self.activation(weighted_sum)

                # Update weights and bias
                error = target - prediction
                if error != 0:
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
                    error_count += 1

            # Stop if no errors
            if error_count == 0:
                print(f"Converged after {iteration + 1} iterations.")
                return
        print("Max iterations reached without convergence.")

    def predict(self, X):
        # Predict for new inputs
        weighted_sum = np.dot(X, self.weights) + self.bias
        return self.activation(weighted_sum)

# Example usage
if __name__ == "__main__":
    # Simple AND dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND output

    perceptron = Perceptron(input_size=2, learning_rate=0.1, max_iter=100)
    perceptron.train(X, y)

    print("\nTesting the Perceptron on AND function:")
    for xi in X:
        print(f"Input: {xi}, Predicted: {perceptron.predict(xi)}")
