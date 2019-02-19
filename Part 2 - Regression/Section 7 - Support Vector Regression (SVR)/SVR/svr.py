# SVR

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:3].values

# Split the data into training set and test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)"""

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# Fitting the SVR Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(X, Y)

# Predicting new data
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))


# Visualizing the SVR
plt.scatter(X, Y, color="red")
plt.plot(X, regressor.predict(X), color="blue")
plt.title("Salary by Level (SVR)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()