# Part 1: K-Nearest Neightbor (KNN) implementation

## KNN implementation for 1 feature
1. Consider the (xi, yi) pairs: [ (7, No), (12, No), (15, Yes), (11, Yes), (4, No), (17, Yes), (9, No), (5, No), (18, Yes), (21, Yes) ]. Note that x is single dimensional.
2. Implement KNN algorithm in python to determine which class does the following test data points belong to for k = 5.<br>
   a. x = 10 <br>
   b. x = 13 <br>
   c. x = 15
3. Determine the class for the above 3 test data points for K = 5.
4. Write down your observations from the above experiments.

## KNN implementation for 2 features
5. Extend the above implementation to accommodate 2 features. Each data point includes (x1, x2, y)
6. Generate synthetic dataset for x1, x2 and y of size 100 values randomly as follows:
   a. x1 between the range 1 and 99.
   b. x2 between the range 1000 and 9999.
   c. y between the range 0 and 1 (2 class).
7. Using matplotlib, plot (x1, x2) for all points that has y = 0 in blue color. Similarly, plot (x1, x2) for all points that has y = 1 in red color.
8. For the following test data points, use your implementation to determine the class (k = 3) (br>
   a. (x1 = 25, x2 = 7835)<br>
   b. (x1 = 99, x2 = 1001)<br>
   c. (x1 = 50, x2 = 5000)<br>
9. Write down your observations based on the above experiments.
10. Standardize both x1 and x2 values between [0-1] range and repeat the predictions. Did you observe any changes to the classification? Write down your observations.

## KNN implementation of multiple features
11. Extend your implementation to deal with 'n' features and try predicting the class with synthetically generated dataset (apply standardization).
12. Use Scikit-learn and TensorFlow to predict the class and compare the preditions against your predictions. Tabulate them.
13. Take a real-world dataset and make predictions using Scikit-learn.
14. KNN is supposed to be computationally intensive for large datasets. Did  you observe this?

# Part 2: Naive Bayes implementation
15. Implement Naive Bayes algorithm (all features are discrete) or download existing implementation and make sure you understand it.
16. Train your algorithm for the example given in [Geeks-for-geeks](https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/) and check its predictions.
17. Repeat the predictions with Scikit-learn and TensorFlow NB classifiers and tabulate them against your implementation.
18. Write down your observations based on the above experiments.
19. (Optional) Modify your implementation to compute likelihood for continuous data (Guassian distribution given mu and sigma). Python libraries can do this. You have to do some prompting and tinkering to get this.
20. (Optional) If features are not independent, Naive Bayes algorithn cannot predict that well. Can you prove this with an appropriate dataset?
