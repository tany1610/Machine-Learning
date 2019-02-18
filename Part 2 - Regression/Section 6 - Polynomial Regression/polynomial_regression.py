# Polynomial Linear Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading the data
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Training the model using linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)

# Training the model using polynomial regression
from sklearn.preprocessing import PolynomialFeatures
regressor_poly = PolynomialFeatures(degree=4)
X_poly = regressor_poly.fit_transform(X)
regressor_poly.fit(X_poly, Y)
regressor2 = LinearRegression()
regressor2.fit(X_poly, Y)

# Visualizing the Linear Regression
plt.scatter(X, Y, color="red")
plt.plot(X, regressor.predict(X), color="blue")
plt.title("Salary by Level (Linear Regression)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Visualizing the Polynomial Regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color="red")
plt.plot(X_grid, regressor2.predict(regressor_poly.fit_transform(X_grid)), color="blue")
plt.title("Salary by Level (Polynomial Regression)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()