# Scikit-Learn

Scikit-learn (often abbreviated as sklearn) is the most popular and comprehensive machine learning library in Python. 
It provides a consistent and simple interface for a vast array of machine learning algorithms for supervised and unsupervised learning, 
along with tools for model selection, preprocessing, and evaluation.

Here are the Scikit-learn essentials you absolutely need to know:

## 1. The Core Scikit-learn API (The Estimator Interface)
One of the most powerful features of Scikit-learn is its consistent API across almost all algorithms. Every estimator (model) follows a common interface,
making it easy to swap out algorithms and understand how to use them.

The common steps are:
1. Instantiate the Estimator: Create an instance of the model you want to use. You often pass hyperparameters during this step.
2. Fit the Model: Train the model using your training data. This is where the learning happens.
3. Predict: Use the trained model to make predictions on new, unseen data.
4. Evaluate: Assess the model's performance.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # An example estimator
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Prepare some dummy data (Features X, Label y)
# Imagine house prices: SqFt, Bedrooms -> Price
X = pd.DataFrame(np.random.rand(100, 2) * 100, columns=['SqFt', 'Bedrooms'])
y = pd.Series(50000 + 100 * X['SqFt'] + 20000 * X['Bedrooms'] + np.random.randn(100) * 10000)

# 2. Split data into training and testing sets
# This is crucial to evaluate how well your model generalizes to unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# 3. Instantiate the Estimator (model)
model = LinearRegression()
print("\nModel instantiated:", model)

# 4. Fit the Model (train the model)
model.fit(X_train, y_train)
print("\nModel trained!")
print("Learned coefficients:", model.coef_)
print("Learned intercept:", model.intercept_)

# 5. Make Predictions
y_pred = model.predict(X_test)
print("\nFirst 5 true prices (test):", y_test.head().tolist())
print("First 5 predicted prices:", y_pred[:5].tolist())

# 6. Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
```

## 2. Data Preprocessing
Raw data is rarely ready for ML models. Scikit-learn provides many tools for cleaning and transforming data.

### a) Scaling Numerical Features

Many ML algorithms (e.g., SVMs, K-Means, Neural Networks, Gradient Descent based models) perform better when numerical input features are scaled to a similar range.
- StandardScaler: Scales data to have zero mean and unit variance (Gaussian distribution with mean 0 and std 1). Best for algorithms that assume normally distributed data or rely on distances.
- MinMaxScaler: Scales data to a fixed range, usually [0, 1]. Good for algorithms that are sensitive to the range of data (e.g., neural networks with sigmoid activation).

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data_to_scale = np.array([[10, 200], [20, 150], [30, 250]])
print("\nOriginal Data for Scaling:\n", data_to_scale)

# StandardScaler
scaler_standard = StandardScaler()
scaled_standard_data = scaler_standard.fit_transform(data_to_scale)
print("\nStandard Scaled Data:\n", scaled_standard_data)
print("Mean (col 0):", scaled_standard_data[:, 0].mean())
print("Std Dev (col 0):", scaled_standard_data[:, 0].std())

# MinMaxScaler
scaler_minmax = MinMaxScaler()
scaled_minmax_data = scaler_minmax.fit_transform(data_to_scale)
print("\nMin-Max Scaled Data:\n", scaled_minmax_data)
print("Min (col 0):", scaled_minmax_data[:, 0].min())
print("Max (col 0):", scaled_minmax_data[:, 0].max())
```

Important Note: Always fit the scaler on the training data only and then transform both the training and test data using the same fitted scaler. This prevents data leakage from the test set.

### b) Encoding Categorical Features

ML models typically require numerical input. Categorical features (like 'City', 'Color') need to be converted.
- OneHotEncoder: Converts categorical variables into a one-hot numerical array. Each category becomes a new binary column (0 or 1).
- LabelEncoder: Converts each category into a unique integer. Useful for target labels (y), but generally avoid for input features (X) as it introduces an artificial ordinal relationship.

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# OneHotEncoder for input features (X)
data_categorical = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
                                 'Size': ['S', 'M', 'L', 'S', 'M']})
print("\nOriginal Categorical Data:\n", data_categorical)

encoder_onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False for dense array
encoded_data = encoder_onehot.fit_transform(data_categorical)
print("\nOne-Hot Encoded Data (Numpy array):\n", encoded_data)
print("Encoded column names:", encoder_onehot.get_feature_names_out(data_categorical.columns))

# LabelEncoder for target labels (y)
labels = ['Dog', 'Cat', 'Dog', 'Bird', 'Cat']
print("\nOriginal Labels:", labels)

encoder_label = LabelEncoder()
encoded_labels = encoder_label.fit_transform(labels)
print("Label Encoded Labels:", encoded_labels)
print("Inverse transformed labels:", encoder_label.inverse_transform([0, 1, 2])) # Map back to original
```

## 3. Model Selection and Evaluation
Choosing the best model and hyperparameters, and accurately assessing performance.

### a) Train-Test Split (train_test_split)
As shown above, this is fundamental for honest model evaluation.

### b) Cross-Validation (KFold, StratifiedKFold, cross_val_score)
A more robust evaluation technique than a single train-test split. It divides the data into 'k' folds, trains on k-1 folds, and tests on the remaining fold, repeating k times. This provides a more reliable estimate of model performance and helps detect overfitting.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression # For classification example

# Prepare some classification data (e.g., Iris dataset)
from sklearn.datasets import load_iris
iris = load_iris()
X_clf, y_clf = iris.data, iris.target

# Instantiate a classifier
clf = LogisticRegression(max_iter=200) # Increase max_iter for convergence

# Perform 5-fold cross-validation
# scoring='accuracy' is default for classification, can be 'f1', 'roc_auc' etc.
scores = cross_val_score(clf, X_clf, y_clf, cv=5, scoring='accuracy')
print(f"\nCross-validation accuracies: {scores}")
print(f"Mean CV accuracy: {scores.mean():.4f}")
print(f"Std Dev of CV accuracy: {scores.std():.4f}")
```

### c) Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV)
Models have hyperparameters that are not learned from data but set by the user (e.g., n_estimators in RandomForest, C in SVM). Tuning them is crucial.

- GridSearchCV: Exhaustively searches over a specified parameter grid for the best combination. Can be computationally expensive.
- RandomizedSearchCV: Samples a fixed number of parameter settings from specified distributions. Often faster than GridSearchCV and can find good results.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200], # Number of trees in the forest
    'max_depth': [None, 10, 20],   # Maximum depth of the tree
    'min_samples_split': [2, 5]    # Minimum number of samples required to split an internal node
}

# Instantiate the model
rf_clf = RandomForestClassifier(random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=1, scoring='accuracy') # n_jobs=-1 uses all CPU cores

# Fit GridSearchCV (this will train many models)
grid_search.fit(X_clf, y_clf)

print(f"\nBest hyperparameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# You can then get the best estimator
best_rf_model = grid_search.best_estimator_
```

### d) Evaluation Metrics
Scikit-learn's metrics module provides a wide range of functions.

- Classification: accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report.
- Regression: mean_absolute_error, mean_squared_error, r2_score.

```python
from sklearn.metrics import classification_report, confusion_matrix

# Assuming `best_rf_model` is trained and `X_test`, `y_test` are available
# (using X_clf, y_clf as test here for simplicity, but normally use separate test set)
y_pred_clf = best_rf_model.predict(X_clf) # Should be X_test here

print("\nConfusion Matrix:\n", confusion_matrix(y_clf, y_pred_clf))
print("\nClassification Report:\n", classification_report(y_clf, y_pred_clf))
```

## 4. Common Estimators (Models)
Scikit-learn offers a vast array of algorithms. Here are some of the most commonly used:

- Regression: LinearRegression, Lasso, Ridge, ElasticNet, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, SVR.
- Classification: LogisticRegression, KNeighborsClassifier, SVC (Support Vector Classifier), DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier (though XGBoost has its own library, sklearn API is available).
- Clustering (Unsupervised): KMeans, DBSCAN, AgglomerativeClustering.
- Dimensionality Reduction (Unsupervised): PCA (Principal Component Analysis), TSNE (t-SNE for visualization).

## 5. Pipelines (Pipeline)
Pipelines allow you to chain multiple processing steps (like scaling, encoding, and then a model) into a single Scikit-learn object. This is excellent for keeping your code clean, avoiding data leakage, and simplifying cross-validation and hyperparameter tuning.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer # For different transformations on different columns

# Sample data with numerical and categorical features
data_pipeline = pd.DataFrame({
    'Numerical_Feature_1': np.random.rand(100) * 100,
    'Numerical_Feature_2': np.random.randn(100) * 10,
    'Categorical_Feature_A': np.random.choice(['X', 'Y', 'Z'], 100),
    'Categorical_Feature_B': np.random.choice(['P', 'Q'], 100)
})
target_pipeline = np.random.randint(0, 2, 100) # Binary target

numerical_features = ['Numerical_Feature_1', 'Numerical_Feature_2']
categorical_features = ['Categorical_Feature_A', 'Categorical_Feature_B']

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(max_iter=500))])

# Split data
X_train_pipe, X_test_pipe, y_train_pipe, y_test_pipe = train_test_split(
    data_pipeline, target_pipeline, test_size=0.2, random_state=42
)

# Fit the pipeline (preprocessing and model training happen in one go)
pipeline.fit(X_train_pipe, y_train_pipe)

# Make predictions
y_pred_pipe = pipeline.predict(X_test_pipe)

print("\nPipeline trained successfully!")
print(f"Pipeline accuracy on test set: {pipeline.score(X_test_pipe, y_test_pipe):.4f}")
```

Mastering these Scikit-learn essentials will equip you with the fundamental skills to build, train, evaluate, and deploy a wide range of machine learning models effectively.
