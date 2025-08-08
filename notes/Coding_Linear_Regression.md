# Linear Regression: Step-by-step implementation


## 1. Basic plot
This is a basic program which plots a set of (age, salary) data in 2D space using Matplotlib.

```python {highlight=10}
import matplotlib.pyplot as plt
import numpy as np

# Define the plot area and set the range
f, ax = plt.subplots() 
ax.set_xlim(left=0, right=100)
ax.set_ylim(bottom=0, top=140000)

# Data points
age = [24, 38, 45, 29, 54, 33]
salary = [35000, 79000, 108000, 64000, 98000, 86000]

# Plotting points
plt.plot(age, salary, 'o', label='Points')  # 'o' specifies circular markers
plt.title("Linear Regression Implementation - Step by step")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()

plt.show()
```
![Basic plot](images/lr01.png)

## 2. Initialization step
Set initial values for **slope m** and **y-intercept c** and draw a line.

```python
import matplotlib.pyplot as plt
import numpy as np

# Data points
...

# Define the plot area and set the range
...

# Initialize slope & y-intercept and draw a line
m=0
c=5000
ax.axline((0, c), slope=m, color='blue', label=f'y = {m}x + {c}')

plt.show()
```
![Initialization of m and c](images/lr02.png)

## 3. Defining error/loss function
Determine the error with respect to the line using Sum of Squares Error.

```python
import matplotlib.pyplot as plt
import numpy as np

# Error function definition
def error(m, c, x, y):
    totalError = 0
    for i in range(0, len(x)):
        yp = m*x[i] + c
        totalError += (y[i] - yp)**2
    return totalError/float(len(x))

# Data points
...

# Define the plot area and set the range
...

# Initialize slope & y-intercept and draw a line
...

# Call error function to compute the error
print("MSE for slope ", m, " and y-intercept ", c ," is ", error(m, c, age, salary))

plt.show()
```
**Output**:  
*MSE for slope  0  and y-intercept  5000  is  5946000000.0*

## 4. Defining the step gradient function
Compute the step gradient for m and c.

```python
import matplotlib.pyplot as plt
import numpy as np

# Error function definition
...

# Step gradient function definition
def step_gradient(m_current, c_current, x, y, learning_rate):
  for i in range(0, len(x)):
    m_gradient = 0
    c_gradient = 0
    N = len(age)
    for i in range(0, len(x)):
        m_gradient += -(2/N)* (y[i] - (m_current * x[i] + c_current)) * x[i]
        c_gradient += -(2/N)* (y[i] - (m_current * x[i] + c_current))
    new_m = m_current - (learning_rate * m_gradient)
    new_c = c_current - (learning_rate * c_gradient)
    return [new_m, new_c]

# Data points
...

# Define the plot area and set the range
...

# Initialize slope & y-intercept and draw a line
...

# Call error function to compute the error
...

# Call step gradient and compute the new m and c and plot the line
learning_rate = 0.0001
m, c = step_gradient(m, c, age, salary, learning_rate)
ax.axline((0, c), slope=m, color='blue', label=f'y = {m}x + {c}')
print("MSE for slope ", m, " and y-intercept ", c ," is ", error(m, c, age, salary))

plt.show()
```
**Output**  
*MSE for slope  0  and y-intercept  5000  is  5946000000.0*  
*MSE for slope  585.7666666666667  and y-intercept  5014.666666666667  is  3021712330.0479627*  
![Updating m and c](images/lr04.png)

## 5. Determine the best fitting line (m & c) by gradient descent
Iterate repeatedly to determine the new values of m and c and draw the lines. You can note that the gap reduces as you go closer.

```python
import matplotlib.pyplot as plt
import numpy as np

# Error function definition
...

# Step gradient function definition
...

# Data points
...

# Define the plot area and set the range
...

# Initialize slope & y-intercept and draw a line
...

# Call error function to compute the error
...

# Call step gradient and compute the new m and c and plot the line
learning_rate = 0.0001
num_iterations = 1000
for i in range(0, num_iterations):
  m, c = step_gradient(m, c, age, salary, learning_rate)
  ax.axline((0, c), slope=m, color='blue', label=f'y = {m}x + {c}')
  print("MSE for slope ", m, " and y-intercept ", c ," is ", error(m, c, age, salary))

plt.show()
```
**Output**  
MSE for slope  0  and y-intercept  5000  is  5946000000.0  
MSE for slope  585.7666666666667  and y-intercept  5014.666666666667  is  3021712330.0479627  
MSE for slope  997.8225966666666  and y-intercept  5024.976201111112  is  1574665066.852526  
MSE for slope  1287.6822500062963  and y-intercept  5032.220724569001  is  858611783.7046785  
MSE for slope  1491.5833131268007  and y-intercept  5037.30917569904  is  504281749.1369992  
MSE for slope  1635.0170736870916  and y-intercept  5040.880944569657  is  328945950.0005138  
MSE for slope  1735.9152985940595  and y-intercept  5043.385808133003  is  242183214.5069467  
MSE for slope  1806.8920334269449  and y-intercept  5045.140160585161  is  199249754.20876214  
MSE for slope  1856.820589259963  and y-intercept  5046.366568437903  is  178004663.7732387  
:  
:  
:  
MSE for slope  1975.8908103045349  and y-intercept  5023.926286687866  is  157187259.57852772  
MSE for slope  1975.8914517595676  and y-intercept  5023.900713073932  is  157187253.0343601  
MSE for slope  1975.8920932059075  and y-intercept  5023.875139806571  is  157187246.49036995  
MSE for slope  1975.8927346435546  and y-intercept  5023.849566885779  is  157187239.94655704  
MSE for slope  1975.8933760725088  and y-intercept  5023.823994311551  is  157187233.40292156  
MSE for slope  1975.8940174927704  and y-intercept  5023.798422083883  is  157187226.85946345  
![Iterative computation of m & c](images/lr05.png)
![Final computation of m & c](images/lr05b.png)
