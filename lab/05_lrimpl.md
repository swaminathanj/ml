# Linear Regression - Implementation & Analysis

## Step-by-step implementation
1. Implement the simple linear regression algorithm by following the steps provided in [Coding step-by-step](Coding_Linear_Regression.md) and get a hang on how the implementation works.

## Playing around with data
2. Add some more data points to the age & salary lists and try the program.
3. Generate data points using random number generator and try the program.
4. Come up with data points such that the program will underfit.
5. Create the data set which exactly obeys y = 2x + 1 equation. Suppose you were to start with m=0, c=0, does your program yield 2 & 1 respectively?
6. Load the data set from a CSV file for a real example and see how it works.

## Playing around with m and c
7. Does starting with a different inital values for m and c matter? Check if you are able to get the same (very close to) final m and c. Prove by experiment.
8. Suppose you updated only m (but not c), how would the gradient descent work. What is final value of c? How does it compare with original experiment?
9. Suppose you updated only c (but not n), how would the gradient descent work. What is final value of m? How does it compare with original experiment?
10. 

# Playing around with the running rate
12. Observe the behavior of the program for different values of running_rate?
13. If running rate is set high, you may not converge. Can you substantiate this with experiments?
14. If running rate is too low, you will take longer time to converge. Can you substantiate this with experiments?
15. Does the running rate have any influence of number of iterations? Substantiate with experiments.

## Playing around with different distance measures
16. The program uses Mean Squared Error (MSE) to implement the error function. Replace it with Mean Absolute Error (MAE) to determine how your program behaves?
17. Implement Sum of Squared Error to observe if there are any changes in your program behavior.
18. Implement Root Mean Square Errot to observe if theere an any cnages in your program behavior.
