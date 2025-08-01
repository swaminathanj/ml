# Pandas

Pandas is another cornerstone library in Python for data manipulation and analysis, built on top of NumPy. While NumPy provides the ndarray for numerical operations, 
Pandas introduces two powerful data structures that are much more suitable for working with structured, tabular data (like spreadsheets or database tables): Series and DataFrame.

For machine learning, Pandas is your go-to tool for:
- Loading data: From various file formats (CSV, Excel, SQL, JSON, etc.).
- Exploring data: Understanding its structure, contents, and basic statistics.
- Cleaning data: Handling missing values, duplicates, and inconsistent formats.
- Transforming data: Reshaping, creating new features, and encoding categorical variables.
- Preparing data: For input into machine learning models.

Here are the Pandas essentials:

## 1. Core Data Structures
### a) Series
What it is: A one-dimensional labeled array capable of holding any data type (integers, strings, floats, Python objects, etc.). 
Think of it as a single column in a spreadsheet or a NumPy array with an associated index (labels for rows).

```python
import pandas as pd
import numpy as np

# Creating a Series from a list
s = pd.Series([10, 20, 30, 40, 50])
print("Series from list:\n", s)

# Creating a Series with custom index labels
s_custom_index = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print("\nSeries with custom index:\n", s_custom_index)

# Accessing elements
print("Element at index 1:", s[1])
print("Element at label 'b':", s_custom_index['b'])

# Series from a dictionary
data_dict = {'apple': 3, 'banana': 1, 'cherry': 5}
s_dict = pd.Series(data_dict)
print("\nSeries from dictionary:\n", s_dict)
```

### b) DataFrame
What it is: A two-dimensional labeled data structure with columns of potentially different types. 
It's the most commonly used Pandas object and can be thought of as a spreadsheet or a SQL table. 
Each column in a DataFrame is essentially a Series.

```python
# Creating a DataFrame from a dictionary of lists/Series
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}
df = pd.DataFrame(data)
print("DataFrame from dictionary:\n", df)

# Creating a DataFrame from a list of dictionaries
data_list_dict = [
    {'Name': 'Eve', 'Age': 22, 'City': 'Miami'},
    {'Name': 'Frank', 'Age': 40, 'City': 'Seattle'}
]
df2 = pd.DataFrame(data_list_dict)
print("\nDataFrame from list of dictionaries:\n", df2)
```

## 2. Reading and Writing Data
This is how you get your data into and out of Pandas. CSV files are the most common.

```python
# Assuming you have a file named 'data.csv'
# Example content for 'data.csv':
# Name,Age,City
# Alice,25,New York
# Bob,30,Los Angeles
# Charlie,35,Chicago

# Read from CSV
try:
    df_csv = pd.read_csv('data.csv')
    print("\nDataFrame from CSV:\n", df_csv)
except FileNotFoundError:
    print("\n'data.csv' not found. Please create it for this example.")
    # Create a dummy CSV for demonstration if it doesn't exist
    df.to_csv('data.csv', index=False)
    df_csv = pd.read_csv('data.csv')
    print("\nCreated and loaded 'data.csv':\n", df_csv)


# Write to CSV
df.to_csv('output.csv', index=False) # index=False prevents writing the DataFrame index as a column
print("\nDataFrame saved to 'output.csv'")

# Other common read/write functions:
# pd.read_excel(), df.to_excel()
# pd.read_sql(), df.to_sql()
# pd.read_json(), df.to_json()
```

## 3. Data Inspection and Exploration
Getting a quick overview of your data.

```python
# First few rows
print("\nHead of DataFrame (first 3 rows):\n", df.head(3))

# Last few rows
print("\nTail of DataFrame (last 2 rows):\n", df.tail(2))

# General information (columns, non-null counts, dtypes, memory usage)
print("\nDataFrame Info:")
df.info()

# Descriptive statistics for numerical columns
print("\nDescriptive Statistics:\n", df.describe())

# Number of rows and columns
print("\nShape of DataFrame (rows, columns):", df.shape)

# Column names
print("Column names:", df.columns)

# Data types of columns
print("Data types:\n", df.dtypes)
```

## 4. Selection and Indexing
Crucial for accessing specific parts of your DataFrame.

### a) Column Selection

```python
# Select a single column (returns a Series)
ages = df['Age']
print("\n'Age' column (Series):\n", ages)

# Select multiple columns (returns a DataFrame)
name_city = df[['Name', 'City']]
print("\n'Name' and 'City' columns (DataFrame):\n", name_city)
```

### b) Row Selection
- .loc[] (label-based indexing)
- .iloc[] (integer-location based indexing)

```python
# Select row by integer position (iloc)
first_row = df.iloc[0]
print("\nFirst row (by position):\n", first_row)

# Select rows by label (loc - if using custom labels, otherwise default integers)
# If your index is default (0, 1, 2...), loc works like iloc for integers.
# However, loc includes the end label in slicing.
rows_0_to_2_loc = df.loc[0:2] # Includes rows 0, 1, 2
print("\nRows 0 to 2 (inclusive, by label with loc):\n", rows_0_to_2_loc)

# Select specific rows and columns by integer position
element_at_0_1 = df.iloc[0, 1] # Row 0, Column 1 (Age of Alice)
print("\nElement at [0, 1]:", element_at_0_1)

# Select specific rows and columns by labels
alice_city = df.loc[0, 'City'] # City of Alice
print("City of Alice (loc):", alice_city)

# Select multiple rows and columns by labels
subset_data = df.loc[1:3, ['Name', 'Age']] # Rows 1 to 3, Name and Age columns
print("\nSubset data (loc):\n", subset_data)

# Boolean indexing (filtering rows based on a condition)
old_enough = df[df['Age'] > 28]
print("\nPeople older than 28:\n", old_enough)

# Multiple conditions
young_people_in_ny = df[(df['Age'] < 30) & (df['City'] == 'New York')]
print("\nYoung people in New York:\n", young_people_in_ny)
```

## 5. Handling Missing Data (NaN - Not a Number)
Real-world data is often messy and has missing values.

```python
df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})
print("\nDataFrame with missing values:\n", df_missing)

# Check for missing values
print("\nMissing values (boolean):\n", df_missing.isnull())

# Count missing values per column
print("\nMissing values count per column:\n", df_missing.isnull().sum())

# Drop rows with any missing values
df_dropped = df_missing.dropna()
print("\nDataFrame after dropping rows with NaN:\n", df_dropped)

# Drop columns with any missing values
df_dropped_cols = df_missing.dropna(axis=1) # axis=1 for columns
print("\nDataFrame after dropping columns with NaN:\n", df_dropped_cols)

# Fill missing values
# Fill with a specific value
df_filled_zero = df_missing.fillna(0)
print("\nDataFrame after filling NaN with 0:\n", df_filled_zero)

# Fill with the mean of the column (common imputation strategy)
df_filled_mean = df_missing.fillna(df_missing['A'].mean())
print("\nDataFrame after filling NaN in 'A' with mean of 'A':\n", df_filled_mean)

# Fill using forward fill (propagate last valid observation forward)
df_ffill = df_missing.fillna(method='ffill')
print("\nDataFrame after forward fill:\n", df_ffill)

# Fill using backward fill (propagate next valid observation backward)
df_bfill = df_missing.fillna(method='bfill')
print("\nDataFrame after backward fill:\n", df_bfill)
```

## 6. Data Manipulation and Transformation
### a) Adding/Modifying Columns

```python
df_copy = df.copy() # Make a copy to avoid modifying original df

# Add a new column
df_copy['Country'] = 'USA'
print("\nDataFrame with new 'Country' column:\n", df_copy)

# Create a new column based on existing ones
df_copy['Age_Category'] = df_copy['Age'].apply(lambda x: 'Adult' if x >= 18 else 'Minor')
print("\nDataFrame with 'Age_Category' column:\n", df_copy)

# Modify an existing column
df_copy['Age'] = df_copy['Age'] + 1 # Increment all ages by 1
print("\nDataFrame with incremented 'Age' column:\n", df_copy)
```

### b) Removing Columns/Rows

```python
# Drop a column (original df remains unchanged unless inplace=True)
df_no_city = df.drop('City', axis=1) # axis=1 for column
print("\nDataFrame after dropping 'City' column:\n", df_no_city)

# Drop a row by index label
df_no_bob = df.drop(1) # Drops the row with index 1 (Bob)
print("\nDataFrame after dropping Bob's row:\n", df_no_bob)
```

### c) Applying Functions

```python
# Apply a function to a Series
df['Age_Times_Two'] = df['Age'].apply(lambda x: x * 2)
print("\n'Age_Times_Two' column:\n", df['Age_Times_Two'])

# Apply a function to the entire DataFrame (more advanced, often for row/column operations)
# df.apply(np.mean, axis=0) # Mean of each column
```

### d) Grouping and Aggregation (groupby())
Powerful for analyzing subsets of data.

```python
df_sales = pd.DataFrame({
    'Product': ['A', 'B', 'A', 'C', 'B', 'A'],
    'Region': ['East', 'West', 'East', 'North', 'East', 'West'],
    'Sales': [100, 150, 120, 80, 200, 90]
})
print("\nSales DataFrame:\n", df_sales)

# Group by 'Product' and calculate sum of 'Sales'
product_sales = df_sales.groupby('Product')['Sales'].sum()
print("\nTotal Sales by Product:\n", product_sales)

# Group by multiple columns and get mean sales
region_product_mean_sales = df_sales.groupby(['Region', 'Product'])['Sales'].mean()
print("\nMean Sales by Region and Product:\n", region_product_mean_sales)

# Apply multiple aggregation functions
agg_results = df_sales.groupby('Region').agg(
    total_sales=('Sales', 'sum'),
    average_sales=('Sales', 'mean'),
    num_transactions=('Sales', 'count')
)
print("\nAggregated Sales by Region:\n", agg_results)
```

### e) Merging and Concatenating DataFrames
Combining multiple DataFrames.

```python
df_customers = pd.DataFrame({
    'CustomerID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David']
})

df_orders = pd.DataFrame({
    'OrderID': [101, 102, 103, 104],
    'CustomerID': [2, 4, 1, 3],
    'Amount': [50.0, 75.5, 120.0, 30.0]
})

# Merge DataFrames (like SQL JOIN)
merged_df = pd.merge(df_customers, df_orders, on='CustomerID', how='inner')
print("\nMerged DataFrame (inner join on CustomerID):\n", merged_df)

# Concatenate DataFrames (stacking them vertically or horizontally)
df_more_data = pd.DataFrame({'Name': ['Eve', 'Frank'], 'Age': [22, 40], 'City': ['Miami', 'Seattle']})
concatenated_df = pd.concat([df, df_more_data], ignore_index=True) # Vertical concatenation
print("\nConcatenated DataFrame (vertical):\n", concatenated_df)
```

## 7. Handling Duplicates

```python
df_dupes = pd.DataFrame({
    'col1': [1, 2, 2, 3, 4, 4],
    'col2': ['A', 'B', 'B', 'C', 'D', 'D']
})
print("\nDataFrame with duplicates:\n", df_dupes)

# Check for duplicates
print("\nDuplicated rows (boolean):\n", df_dupes.duplicated())

# Drop duplicate rows (keeps first occurrence by default)
df_no_dupes = df_dupes.drop_duplicates()
print("\nDataFrame after dropping duplicates:\n", df_no_dupes)

# Drop duplicates based on specific column(s)
df_unique_col1 = df_dupes.drop_duplicates(subset=['col1'])
print("\nDataFrame unique based on 'col1':\n", df_unique_col1)
```

## 8. Data Type Conversion
Ensuring your data has the correct type for operations or modeling.

```python
df_dtypes = pd.DataFrame({'Numbers': ['1', '2', '3'], 'Text': ['a', 'b', 'c']})
print("\nOriginal Data Types:\n", df_dtypes.dtypes)

# Convert a column to numeric
df_dtypes['Numbers'] = pd.to_numeric(df_dtypes['Numbers'])
print("\nData Types after converting 'Numbers' to numeric:\n", df_dtypes.dtypes)

# Convert to datetime (crucial for time series)
date_data = pd.Series(['2023-01-01', '2023-01-02', '2023-01-03'])
dates = pd.to_datetime(date_data)
print("\nConverted to Datetime Series:\n", dates)
print("Datetime Series Dtype:", dates.dtype)
```

Mastering these Pandas essentials will make you highly proficient in the data wrangling and preprocessing steps, which are often the most time-consuming parts of any machine learning project.
