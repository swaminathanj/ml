# Linear Regression - Evaluation, Tuning, Regularization

## Plain linear regression code
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from google.colab import files # Import files module for uploading
from sklearn.metrics import r2_score


class LinearRegression():
    def __init__(self, learning_rate, iterations, l1_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)

        dW = np.zeros(self.n)
        for j in range(self.n):
            if self.W[j] > 0:
                # Gradient for positive weights
                dW[j] = (-2 * (self.X[:, j]).dot(self.Y - Y_pred)) / self.m # + self.l1_penalty) / self.m
            else:
                # Gradient for non-positive weights
                dW[j] = (-2 * (self.X[:, j]).dot(self.Y - Y_pred)) / self.m # + self.l1_penalty) / self.m

        db = -2 * np.sum(self.Y - Y_pred) / self.m

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def predict(self, X):
        # Linear model prediction
        return X.dot(self.W) + self.b

def runalgo():
    # Upload the file if it's not found
    try:
        df = pd.read_csv("Experience-Salary.csv")
    except FileNotFoundError:
        print("Uploading 'Experience-Salary.csv'...")
        uploaded = files.upload()
        for fn in uploaded.keys():
            print('User uploaded file "{name}" with length {length} bytes'.format(
                name=fn, length=len(uploaded[fn])))
        df = pd.read_csv("Experience-Salary.csv") # Try reading again after upload

    # Separate features (X) and target (Y)
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1/3, random_state=0)

    # Initialize and train the Lasso Regression model
    model2 = LinearRegression(
        iterations=1000, learning_rate=0.01, l1_penalty=500)
    model2.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = model2.predict(X_test)

    # Print results
    print("Predicted values: ", np.round(Y_pred[:3], 2))
    print("Real values:      ", Y_test[:3])
    print("Trained W:        ", round(model2.W[0], 2))
    print("Trained b:        ", round(model2.b, 2))
    r2 = r2_score(Y_test, Y_pred)
    print(f"R-squared: {r2}")
    adjusted_r2 = 1 - (1 - r2) * (Y_test.size - 1) / (Y_test.size - 1 - 1)
    print(f"Adjusted R-squared: {adjusted_r2}")

    # Visualize the results
    plt.scatter(X_test, Y_test, color='blue', label='Actual Data')
    plt.plot(X_test, Y_pred, color='orange', label='Linear Regression Line')
    plt.title('Salary vs Experience (Linear Regression)')
    plt.xlabel('Years of Experience (Standardized)')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()


#if __name__ == "__main__":
runalgo()
```

## Lasso regression code
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from google.colab import files # Import files module for uploading
from sklearn.metrics import r2_score


class LassoRegression():
    def __init__(self, learning_rate, iterations, l1_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)

        dW = np.zeros(self.n)
        for j in range(self.n):
            if self.W[j] > 0:
                # Gradient for positive weights
                dW[j] = (-2 * (self.X[:, j]).dot(self.Y - Y_pred) +
                         self.l1_penalty) / self.m
            else:
                # Gradient for non-positive weights
                dW[j] = (-2 * (self.X[:, j]).dot(self.Y - Y_pred) -
                         self.l1_penalty) / self.m

        db = -2 * np.sum(self.Y - Y_pred) / self.m

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def predict(self, X):
        # Linear model prediction
        return X.dot(self.W) + self.b

def runalgo():
    # Upload the file if it's not found
    try:
        df = pd.read_csv("Experience-Salary.csv")
    except FileNotFoundError:
        print("Uploading 'Experience-Salary.csv'...")
        uploaded = files.upload()
        for fn in uploaded.keys():
            print('User uploaded file "{name}" with length {length} bytes'.format(
                name=fn, length=len(uploaded[fn])))
        df = pd.read_csv("Experience-Salary.csv") # Try reading again after upload

    # Separate features (X) and target (Y)
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1/3, random_state=0)

    # Initialize and train the Lasso Regression model
    model = LassoRegression(
        iterations=1000, learning_rate=0.01, l1_penalty=500)
    model.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = model.predict(X_test)

    # Print results
    print("Predicted values: ", np.round(Y_pred[:3], 2))
    print("Real values:      ", Y_test[:3])
    print("Trained W:        ", round(model.W[0], 2))
    print("Trained b:        ", round(model.b, 2))
    r2 = r2_score(Y_test, Y_pred)
    print(f"R-squared: {r2}")
    adjusted_r2 = 1 - (1 - r2) * (Y_test.size - 1) / (Y_test.size - 1 - 1)
    print(f"Adjusted R-squared: {adjusted_r2}")

    # Visualize the results
    plt.scatter(X_test, Y_test, color='blue', label='Actual Data')
    plt.plot(X_test, Y_pred, color='orange', label='Lasso Regression Line')
    plt.title('Salary vs Experience (Lasso Regression)')
    plt.xlabel('Years of Experience (Standardized)')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()


#if __name__ == "__main__":
runalgo()
```

## Data sets
1. Create a dataset 


## R2 and Adjusted R2
