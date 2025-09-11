# Part 1: K-Nearest Neightbor (KNN) implementation

## KNN implementation for 1 feature
1. Consider the (xi, yi) pairs: [ (7, No), (12, No), (15, Yes), (11, Yes), (4, No), (17, Yes), (9, No), (5, No), (18, Yes), (21, Yes) ]. Note that x is single dimensional.
2. Implement KNN algorithm in python to determine which class does the following test data points belong to for k = 5.
   (i) x = 10,
   (ii) x = 13,
   (iii) x = 15
4. Determine the class for the above 3 test data points for K = 5.
5. Write down your observations from the above experiments.

## KNN implementation for 2 features
1. Extend the above implementation to accommodate 2 features. 
2. Each data point includes (x1, x2, y)
3. Generate 100 values randomly for x1 between the range 1 to 99.
4. Generate 100 values randomly for x2 between the range 1000 to 9999.
5. Generate 100 values randomly for y with either 0 or 1 (2 class).
6. Using matplotlib, plot the points (x1, x2) for all points that has y = 0 in blue color.
7. Similarly, plost the points (x1, x2) for all points that has y = 1 in red color.
8. For a test data point (x1 = 25, x2 = 7835), use the implementation to determine the class.

# Part 2: Naive Bayes implementation
