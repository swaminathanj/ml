

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred):
    # binary cross entropy
    epsilon = 1e-9
    y1 = y_true * np.log(y_pred + epsilon)
    y2 = (1-y_true) * np.log(1 - y_pred + epsilon)
    return -(y1+y2) / len(y_true) # -np.mean(y1 + y2)

def train_logistic_regression(X, y, learning_rate, num_iterations):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    weights_hist = []
    bias_hist = []

    for _ in range(num_iterations):
        linear_model = np.dot(X, weights) + bias
        y_predicted = sigmoid(linear_model)

        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / n_samples) * np.sum(y_predicted - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        weights_hist.append(weights.copy())
        bias_hist.append(bias)

    return weights_hist, bias_hist


# Assuming X, y, learning_rate, num_iterations are defined
# Example: Generate some dummy data for demonstration
#X_dummy = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
#y_dummy = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

X_values = np.array([ [20], [25], [30], [35], [40], [45], [50], [55], [60], [65], [70], [75], [80], [85] ])
y_values = np.array([ 0,  0,  0,  0,  0,  1,  1,  1,  1,  1, 1, 1, 1, 1 ])

weights_hist, bias_hist = train_logistic_regression(X_values, y_values, learning_rate=0.1, num_iterations=100000)

plt.figure(figsize=(10, 6))
x_plot = np.linspace(X_values.min() - 1, X_values.max() + 1, 100) # Range for plotting sigmoid

# Plot initial sigmoid (optional)
initial_linear_model = np.dot(x_plot.reshape(-1, 1), weights_hist[0]) + bias_hist[0]
plt.plot(x_plot, sigmoid(initial_linear_model), label=f'Iteration 0', linestyle='--')

# Plot sigmoid curves at chosen intervals
plot_iterations = [10000, 50000, 99999] # Example: plot at specific iterations
for i in plot_iterations:
    current_weights = weights_hist[i]
    current_bias = bias_hist[i]
    current_linear_model = np.dot(x_plot.reshape(-1, 1), current_weights) + current_bias
    plt.plot(x_plot, sigmoid(current_linear_model), label=f'Iteration {i+1}')

plt.scatter(X_values, y_values, color='red', marker='o', label='Data Points')
plt.xlabel('X')
plt.ylabel('Probability')
plt.title('Sigmoid Curves during Logistic Regression Training')
plt.legend()
plt.grid(True)
plt.show()
```
