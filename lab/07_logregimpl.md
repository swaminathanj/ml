# Logistic Regression - Lab activity

## Understand the code
1. Check out the code in ([Logistic Regression Implmentation](../notes/Coding_Logistic_Regression.md)) and understand the its connection with the math behind.
2. Run the code in Google Colab and understand its working.

## Play around with data (keep X single dimensional)
3. Change the data points and see its effect on the output.
4. Generate data points using random number generator and try the program.
5. Try to come up with data points such that the program will perfectly fit.
6. Try to come up with data points such that the program will underfit.
7. Load the data set from a CSV file for a real example and see how it works.

## Play around with training parameters (keep X single dimensional)
8. Does starting with a different inital values for w and b matter? Check if you are able to get the same (very close to) final w and c. Prove by experiment.
9. Suppose you updated only w (but not b), how would the gradient descent work. What is final value of w? How does it compare with original experiment?
10. Suppose you updated only b (but not w), how would the gradient descent work. What is final value of b? How does it compare with original experiment?
    
## Play around with hyperparameter running rate (keep X single dimensional)
11. Observe the behavior of the program for different values of running_rate.
12. Does the running rate have any influence of number of iterations? Substantiate with experiments.

## Play around with epsilon & number of iterations (keep X single dimensional)
13. Remove epsilon or make it 0 and print error during each iteration to see the effect. Do you get Not-a-number (nan) at any stage?
14. Do different number of iterations (100, 1000, 10000, 100000) and plot the sigmoid at specific intervals to see how it improves as iterations increase.

## Play around with test datasets (X will be multi-dimensional)
15. Try with at least 3 different binary classification datasets from pandas, scikit-learn or kaggle (say disease diagnosis, spam detection, weather forecast).
16. **Note**: Visualization with matplotlib will not work for multi-dimensional data. You have to remove/comment out those statements and print only the parameters, cost, etc.
17. Ensure you divide the data sets into training and testing. Compute and print the different classification metrics for each of the 3 data sets.

## Compute performance with different classification metrics
18. Accuracy with 3 different thresholds (0.5, 0.25, 0.75). Record your observations for the 3 datasets.
19. Precision with 3 different thresholds (0.5, 0.25, 0.75). Record your observations for the 3 datasets.
20. Recall with 3 different thresholds (0.5, 0.25, 0.75). Record your observations for the 3 datasets.
21. F1 score with 3 different thresholds (0.5, 0.25, 0.75). Record your observations for the 3 datasets.
22. Confusion matrix with 3 different thresholds (0.5, 0.25, 0.75). Record your observations for the 3 datasets.
23. AUC ROC curve with 3 different thresholds (0.5, 0.25, 0.75). Record your observations for the 3 datasets.

## Use ML libraries
24. Determine the final parameters and error for a single data set by using Scikit-learn and compare against your implementation.
25. Determine the final parameters and error for a single data set by using Tensor Flow and compare against your implementation.
    (Note: You can tabulate them) - weights, bias, error as rows and the your implementation, Scikit-lear and Tensor Flow as columns.

