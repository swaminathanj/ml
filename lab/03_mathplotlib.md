# Math Plot Lib

Matplotlib is the foundational plotting library in Python. While other libraries like Seaborn are built on top of it, 
understanding Matplotlib's core concepts is crucial for creating highly customized visualizations and for effective 
debugging when other libraries don't give you exactly what you need.

The most common way to use Matplotlib is through its pyplot interface, which provides a MATLAB-like way of plotting.

Here are the Matplotlib essentials:

## 1. The pyplot Module (and the Standard Import)
You almost always start by importing matplotlib.pyplot, conventionally as plt.

```python
import matplotlib.pyplot as plt
import numpy as np # Often used with Matplotlib
```

## 2. The Figure and Axes Objects
This is the most important concept to grasp for effective Matplotlib usage.

- Figure: The entire window or page that contains the plot(s). Think of it as the canvas on which you draw. You can have multiple Axes (subplots) on a single Figure.
- Axes: The actual plotting area where the data is drawn. It contains the x-axis, y-axis, labels, title, and the plot itself (e.g., lines, bars, points). 

A Figure can contain one or many Axes objects.

While plt.plot() provides a quick way to plot, explicitly creating Figure and Axes objects gives you much more control.

```python
# Create a Figure and a single Axes object
fig, ax = plt.subplots()

# Plot data on the Axes object
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

# Add labels and title using the Axes object
ax.set_xlabel("X-axis Label")
ax.set_ylabel("Y-axis Label")
ax.set_title("My First Matplotlib Plot")

plt.show() # Display the plot
```

## 3. Basic Plot Types

### a) Line Plots (plt.plot())
Used to visualize trends over a continuous range, typically time series or functions.

```python
x = np.linspace(0, 10, 100) # 100 points between 0 and 10
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(figsize=(8, 4)) # Create a new figure with specific size

plt.plot(x, y_sin, label='sin(x)', color='blue', linestyle='-', linewidth=2)
plt.plot(x, y_cos, label='cos(x)', color='red', linestyle='--', marker='o', markersize=4, markevery=10) # Example of marker and markevery

plt.xlabel("X-value")
plt.ylabel("Y-value")
plt.title("Sine and Cosine Waves")
plt.legend() # Show the labels defined with 'label' argument
plt.grid(True) # Add a grid
plt.show()
```

### b) Scatter Plots (plt.scatter())
Used to show the relationship between two variables, often for discrete data points.

```python
np.random.seed(42)
num_points = 50
x_data = np.random.rand(num_points) * 10
y_data = 2 * x_data + np.random.randn(num_points) * 2 + 5
colors = np.random.rand(num_points)
sizes = np.random.rand(num_points) * 100 + 20

plt.figure(figsize=(7, 6))
plt.scatter(x_data, y_data, c=colors, s=sizes, alpha=0.7, cmap='viridis', edgecolors='black')
plt.colorbar(label="Random Color Value") # Add a color bar for 'c' argument
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.title("Scatter Plot with Color and Size Variation")
plt.show()
```

### c) Bar Plots (plt.bar())
Used to compare quantities of different categories.

```python
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 12]

plt.figure(figsize=(6, 5))
plt.bar(categories, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.xlabel("Category")
plt.ylabel("Value")
plt.title("Bar Plot of Categories")
plt.show()
```

### d) Histograms (plt.hist())
Used to visualize the distribution of a single numerical variable.

```python
data = np.random.randn(1000) # 1000 random numbers from a standard normal distribution

plt.figure(figsize=(7, 5))
plt.hist(data, bins=30, color='teal', alpha=0.7, edgecolor='black') # bins control the number of bars
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Random Data")
plt.show()
```

4. Customization Essentials
a) Titles and Labels
plt.title() or ax.set_title()

plt.xlabel(), plt.ylabel() or ax.set_xlabel(), ax.set_ylabel()

b) Legends
plt.legend() or ax.legend(): Requires a label argument in your plot() calls.

c) Colors, Linestyles, Markers
Arguments like color, linestyle (-, --, - ., :), marker (o, x, s, ^), linewidth, markersize in plt.plot() and plt.scatter().

d) Axis Limits and Ticks
plt.xlim(), plt.ylim() or ax.set_xlim(), ax.set_ylim()

plt.xticks(), plt.yticks() or ax.set_xticks(), ax.set_yticks() to set specific tick locations.

plt.xticklabels(), plt.yticklabels() or ax.set_xticklabels(), ax.set_yticklabels() to set custom labels for ticks.

Python

x = np.linspace(0, 2 * np.pi, 50)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, y)

ax.set_xlim(0, 2 * np.pi) # Set x-axis limits
ax.set_ylim(-1.1, 1.1)   # Set y-axis limits

# Set custom x-ticks and labels
tick_locations = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
tick_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'] # LaTeX for Greek letters
ax.set_xticks(tick_locations)
ax.set_xticklabels(tick_labels, fontsize=12)

ax.set_title("Sine Wave with Custom Ticks")
plt.show()
e) Grid
plt.grid(True) or ax.grid(True) to add a grid.

f) Figure Size
plt.figure(figsize=(width, height)) to control the overall size of the plot in inches.

5. Subplots
Creating multiple plots within a single Figure.

plt.subplot(nrows, ncols, index): Older, less flexible.

plt.subplots(nrows, ncols): Recommended. Returns a Figure and an array of Axes objects.

Python

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4)) # 1 row, 2 columns

# Plot on the first Axes (left)
axes[0].plot(x, y_sin, color='blue')
axes[0].set_title("Sine Wave")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Sin(X)")

# Plot on the second Axes (right)
axes[1].plot(x, y_cos, color='red')
axes[1].set_title("Cosine Wave")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Cos(X)")

plt.tight_layout() # Adjust subplot params for a tight layout
plt.show()
For more complex grids of subplots:

Python

fig, axes = plt.subplots(2, 2, figsize=(10, 8)) # 2 rows, 2 columns

# Accessing specific axes: axes[row_index, col_index]
axes[0, 0].plot(x, y_sin, 'g')
axes[0, 0].set_title('Top Left')

axes[0, 1].scatter(x_data, y_data)
axes[0, 1].set_title('Top Right')

axes[1, 0].hist(data, bins=20)
axes[1, 0].set_title('Bottom Left')

axes[1, 1].bar(categories, values)
axes[1, 1].set_title('Bottom Right')

plt.tight_layout()
plt.show()
6. Saving Plots
Python

# Save the last shown plot to a file
plt.savefig("my_plot.png") # PNG is good for web
plt.savefig("my_plot.pdf") # PDF is good for print/vector graphics
plt.savefig("my_plot.svg") # SVG is also vector graphics

# For specific figures:
fig.savefig("my_specific_figure.png", dpi=300) # dpi controls resolution
7. plt.show()
Essential for displaying the plot. In interactive environments (like Jupyter Notebooks or IPython), you might see plots without it, but it's good practice to include it.

It blocks the program execution until the plot window is closed.

Mastering these Matplotlib essentials will give you the power to create a vast array of static, publication-quality visualizations and serve as a strong foundation for using higher-level plotting libraries.
