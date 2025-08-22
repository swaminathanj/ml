# Machine Learning

## 1. Foundations (Prerequisites and Introduction)

### Introduction to Machine Learning
1. What is ML? ([Lecture 1](notes/Lecture-1.pdf))
2. Supervised vs. Unsupervised vs. Reinforcement Learning ([Lecture 2](notes/Lecture-2.pdf))
3. Applications of Machine Learning ([Activity 1](exercises/Activity-1.pdf))
4. Basic ML Terminology ([Lecture 3](notes/Lecture-3.pdf))

### Python for Machine Learning
1. NumPy for numerical operations ([Numpy essentials](lab/01_numpy.md))
2. Pandas for data manipulation ([Pandas essentials](lab/02_pandas.md))
3. Matplotlib/Seaborn for data visualization ([Matplotlib essentials](lab/03_mathplotlib.md))
4. Scikit-learn basics for loading datasets, splitting data ([Scikit-learn essentials](lab/04_sklearn.md))

### Overview of Mathematical Foundations
1. Linear Algebra ([Importance of LA in ML](notes/Lecture-4.pdf))
2. Probability & Statistics ([Importance of Probability & Statistics in ML](notes/Lecture-5.pdf))
3. Calculus ([Importance of Calculus in ML](notes/Lecture-6.pdf))

## 2. Supervised Learning - Regression

### Data Preprocessing and Exploration
1. Handling missing values ([Reading from geeks-for-geeks](https://www.geeksforgeeks.org/machine-learning/managing-missing-data-in-linear-regression/)]
2. Feature scaling - standardization, normalization - ([Reading from geeks-for-geeks](https://www.geeksforgeeks.org/machine-learning/ml-feature-scaling-part-2/))
3. One-hot encoding for categorical features ([Reading from geeks-for-geeks](https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/))
4. Exploratory Data Analysis EDA ([Reading from geeks-for-geeks](https://www.geeksforgeeks.org/data-analysis/what-is-exploratory-data-analysis/))

### Linear Regression
1. Simple Linear Regression ([Video: Mathematical intuition](https://www.youtube.com/watch?v=OM1dtIt0VNo))
2. Cost function (Mean Squared Error - MSE)
3. Gradient Descent algorithm ([Video: Mathematical intuition](https://youtu.be/9H-s5cQ1iBk?si=um9J605sChHFYH6b))
4. Multiple Linear Regression ([Reading from Geeks-for-geeks](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/))
5. Assumptions of Linear Regression ([Reading from Geeks-for-geeks](https://www.geeksforgeeks.org/machine-learning/assumptions-of-linear-regression/))

### Model Evaluation for Regression
1. R^2, Adjusted R^2 (Videos: [Intuition behind R^2](https://www.youtube.com/watch?v=-7U10N8PvlQ), [Intuition behind Adjusted R^2](https://www.youtube.com/watch?v=IN6YkHtdgZI))
2. MAE, MSE, RMSE
3. Overfitting and Underfitting
4. Bias-Variance Tradeoff ([Intuition behind](https://www.youtube.com/watch?v=EuBBz3bI-aA))
5. Multi-colinearity ([Reading from Geeks-for-geeks](https://www.geeksforgeeks.org/machine-learning/multicollinearity-in-regression-analysis/))
6. Cross-validation (k-fold, leave-one-out) ([Intuition](https://www.youtube.com/watch?v=fSytzGwwBVw))

### Regularization
1. Ridge Regression (L2 regularization) ([Intuition](https://www.youtube.com/watch?v=Q81RR3yKn30), [Reading](https://www.geeksforgeeks.org/machine-learning/implementation-of-ridge-regression-from-scratch-using-python/))
2. Lasso Regression (L1 regularization) ([Intuition][https://www.youtube.com/watch?v=NGf0voTMlcs], [Reading](https://www.geeksforgeeks.org/machine-learning/implementation-of-lasso-regression-from-scratch-using-python/))
3. Elastic Net ([Intuition](https://www.youtube.com/watch?v=1dKRdX9bfIo&t=13s))

### Lab Activity
1. Implementation ([Coding step-by-step](notes/Coding_Linear_Regression.md))
2. Lab Assignment - 1 ([Lab Activity](lab/05_lrimpl.md))
3. Lab Assignment - 2 ([Lab Activity](lab/06_lreval.md))

## 3. Supervised Learning - Classification

### Introduction to Classification
1. Binary vs. Multi-class classification
2. Classification metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix)
3. ROC Curve and AUC

### Logistic Regression
1. Sigmoid function
2. Cost function (Cross-Entropy)
3. Decision boundary

### K-Nearest Neighbors (KNN)
1. Distance metrics
2. Choosing 'k'

### Support Vector Machines (SVM)
1. Linear SVM (maximal margin hyperplane)
2. Kernels (polynomial, RBF) for non-linear separation

### Decision Trees
1. Entropy, Gini impurity
2. Information Gain
3. Pruning

### Ensemble Methods
1. Bagging: Random Forest
2. Boosting: AdaBoost, Gradient Boosting Machines (GBM), XGBoost, LightGBM, CatBoost

## 4. Unsupervised Learning

### Clustering
1. K-Means Clustering: Elbow method, silhouette score
2. Hierarchical Clustering: Dendrograms
3. DBSCAN (briefly).

### Dimensionality Reduction
1. Principal Component Analysis (PCA): Intuition, eigen decomposition (briefly), applications

