# NumPy Essentials

NumPy (Numerical Python) is the foundational library for scientific computing in Python. It provides a high-performance multi-dimensional array object (ndarray) and tools for working with these arrays. 
It's absolutely essential for machine learning because virtually all data in ML is represented and manipulated as NumPy arrays.

Here are the NumPy essentials you need to grasp:

## 1. The ndarray (N-dimensional Array)
This is the core data structure in NumPy. It's a grid of values, all of the same type, and is indexed by a tuple of non-negative integers.

Key characteristics:
* Homogeneous: All elements in a NumPy array must be of the same data type (e.g., all integers, all floats). This is a key difference from Python lists, which can hold elements of different types.
* Fixed Size: Once created, the total size (number of elements) of an array cannot change. You can reshape it, but the total number of elements remains constant.
* Efficient Operations: NumPy operations are highly optimized (often implemented in C or Fortran) making them much faster than equivalent operations on Python lists, especially for large datasets.
* Vectorization: This is the ability to perform operations on entire arrays at once, rather than looping through individual elements. This is crucial for performance.

Common ndarray attributes:
* .shape: A tuple indicating the size of the array in each dimension.
* .ndim: The number of dimensions (axes) of the array.
* .size: The total number of elements in the array.
* .dtype: The data type of the elements in the array (e.g., int32, float64).

## 2. Array Creation
You need to know how to create arrays in various ways:

```python
import numpy as np

# From a Python list or tuple
arr1 = np.array([1, 2, 3])
print("1D Array:", arr1, arr1.shape, arr1.dtype)

arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", arr2, arr2.shape)

# Arrays of zeros, ones, or empty
zeros_arr = np.zeros((2, 3)) # 2 rows, 3 columns of zeros
print("Zeros Array:\n", zeros_arr)

ones_arr = np.ones((3, 2)) # 3 rows, 2 columns of ones
print("Ones Array:\n", ones_arr)

empty_arr = np.empty((2, 2)) # Uninitialized (contains arbitrary values)
print("Empty Array:\n", empty_arr)

# Arrays with a range of numbers
# `arange` is like Python's `range`, but returns an array
range_arr = np.arange(0, 10, 2) # Start, stop (exclusive), step
print("Arange Array:", range_arr)

# `linspace` creates evenly spaced numbers over a specified interval
linspace_arr = np.linspace(0, 1, 5) # Start, stop (inclusive), number of elements
print("Linspace Array:", linspace_arr)

# Random arrays (useful for initializing weights in neural networks or simulations)
rand_arr = np.random.rand(2, 2) # Uniform distribution [0, 1)
print("Random Array (uniform):\n", rand_arr)

randn_arr = np.random.randn(2, 2) # Standard normal distribution
print("Random Array (normal):\n", randn_arr)

randint_arr = np.random.randint(0, 10, size=(2, 3)) # Random integers [low, high)
print("Random Integers:\n", randint_arr)
```

## 3. Indexing and Slicing
Accessing elements and subarrays is fundamental, similar to Python lists but extended for multiple dimensions.

```python
arr = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])

# Accessing a single element (row, column)
print("Element at (0, 1):", arr[0, 1]) # Output: 20

# Slicing rows
print("First row:", arr[0, :]) # Output: [10 20 30]
print("Last row:", arr[-1, :]) # Output: [70 80 90]

# Slicing columns
print("First column:", arr[:, 0]) # Output: [10 40 70]
print("Second column:", arr[:, 1]) # Output: [20 50 80]

# Sub-array (rows 0-1, columns 1-2)
print("Sub-array:\n", arr[0:2, 1:3])

# Boolean indexing (selecting elements based on a condition)
bool_arr = arr > 50
print("Boolean array (elements > 50):\n", bool_arr)
print("Elements > 50:", arr[bool_arr]) # Output: [60 70 80 90]

# You can also combine conditions
print("Elements > 20 and < 80:", arr[(arr > 20) & (arr < 80)])
```

## 4. Basic Array Operations (Element-wise)
NumPy allows you to perform mathematical operations directly on entire arrays, which are applied element by element.

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Addition, Subtraction, Multiplication, Division
print("Addition:", arr1 + arr2)      # [5 7 9]
print("Subtraction:", arr2 - arr1)   # [3 3 3]
print("Multiplication:", arr1 * arr2) # [4 10 18]
print("Division:", arr2 / arr1)      # [4.  2.5 2. ]

# Scalar operations (applied to every element)
print("Scalar Multiplication:", arr1 * 5) # [ 5 10 15]
print("Scalar Addition:", arr1 + 10)     # [11 12 13]

# Trigonometric functions, exponentials, logarithms
print("Sin of arr1:", np.sin(arr1))
print("e^arr1:", np.exp(arr1))
```

## 5. Aggregation Functions
NumPy provides functions to summarize data in arrays (sum, mean, max, min, standard deviation, etc.).

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print("Sum of all elements:", np.sum(arr)) # Output: 21
print("Mean of all elements:", np.mean(arr)) # Output: 3.5

# Aggregation along an axis (0 for columns, 1 for rows)
print("Sum along columns (axis=0):", np.sum(arr, axis=0)) # Output: [5 7 9] (1+4, 2+5, 3+6)
print("Mean along rows (axis=1):", np.mean(arr, axis=1)) # Output: [2.  5.] ((1+2+3)/3, (4+5+6)/3)

print("Max element:", np.max(arr))
print("Min element of each column:", np.min(arr, axis=0))
print("Standard deviation:", np.std(arr))
```

## 6. Reshaping and Transposing
Changing the dimensions or orientation of an array without changing its data.

```python
arr = np.arange(1, 10) # [1 2 3 4 5 6 7 8 9]
print("Original array:", arr)

# Reshape (must match total number of elements)
reshaped_arr = arr.reshape(3, 3)
print("Reshaped to 3x3:\n", reshaped_arr)

# Transpose (swaps rows and columns)
transposed_arr = reshaped_arr.T
print("Transposed array:\n", transposed_arr)

# Flatten an array back to 1D
flattened_arr = reshaped_arr.flatten()
print("Flattened array:", flattened_arr)
```

## 7. Broadcasting
This is a powerful feature that allows NumPy to perform operations on arrays of different shapes, provided they are compatible. NumPy automatically "stretches" the smaller array to match the larger one.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]]) # Shape (2, 3)

vector = np.array([10, 20, 30]) # Shape (3,)

# NumPy broadcasts the vector across each row of the array
result = arr + vector
print("Array + Vector (Broadcasting):\n", result)

# Example with a scalar (most common broadcasting)
result_scalar = arr + 100
print("Array + Scalar:\n", result_scalar)
```

## 8. Linear Algebra Operations
NumPy is incredibly powerful for linear algebra, which is fundamental to many ML algorithms.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiplication (already covered)
print("Element-wise multiplication:\n", A * B)

# Matrix multiplication (use @ operator or np.dot)
print("Matrix multiplication (A @ B):\n", A @ B)
print("Matrix multiplication (np.dot(A, B)):\n", np.dot(A, B))

# Dot product of vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print("Dot product of vectors:", np.dot(v1, v2)) # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

# Inverse of a matrix
try:
    inverse_A = np.linalg.inv(A)
    print("Inverse of A:\n", inverse_A)
    print("A @ A_inverse:\n", A @ inverse_A) # Should be identity matrix
except np.linalg.LinAlgError:
    print("Matrix A is singular (no inverse).")

# Determinant of a matrix
print("Determinant of A:", np.linalg.det(A))
```

These are the core functionalities of NumPy that you'll use constantly in machine learning and data science. Mastering them will significantly boost your productivity and enable you to handle large numerical datasets efficiently.







