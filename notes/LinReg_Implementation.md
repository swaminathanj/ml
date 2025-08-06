# Linear Regression: Step-by-step implementation

## 1. Basic plot

```python
import matplotlib.pyplot as plt
import numpy as np

# Data points
age = [24, 38, 45, 29, 54, 33]
salary = [35000, 79000, 108000, 64000, 98000, 86000]
m=0
c=5000

f, ax = plt.subplots() 
ax.set_xlim(left=0, right=100) 
ax.set_ylim(bottom=0, top=140000)          # Adjust y-axis to start from 0

# Plotting points
plt.plot(age, salary, 'o', label='Points')  # 'o' specifies circular markers
plt.title("Linear Regression Implementation")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()

ax.axline((0, c), slope=m, color='blue', label=f'y = {m}x + {c}')
plt.show()
```

## 2. Initialization step

## 3. Defining error/loss function

## 4. Defining the step function

## 5. Gradient descent

