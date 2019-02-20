# Random Forest Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Split the data into training set and test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)"""

# feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X  = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state=0, n_estimators=300)
regressor.fit(X, Y)

# Predicting new data
y_pred = regressor.predict([[6.5]])


# Visualizing the Random Forest Regression
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Salary by Level (Random Forest Regression)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()