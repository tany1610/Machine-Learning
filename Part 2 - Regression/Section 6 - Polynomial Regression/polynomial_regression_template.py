# Polynomial Regression Template

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


# Fitting the Regression Modelto the dataset


# Predicting new data
y_pred = regressor.predict(6.5)


# Visualizing the Polynomial Regression
plt.scatter(X, Y, color="red")
plt.plot(X, regressor.predict(X), color="blue")
plt.title("Salary by Level (Polynomial Regression)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()