# Artificial Neural Network

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Split the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X  = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting the Classifier to the dataset
import keras
from keras.models import Sequential
from keras.layers import Dense

# Creating the ann
classifier = Sequential()

# Creating the input and hidden layers
classifier.add(Dense(units=6, input_shape=11, activation='relu'))

# Adding one more hidden layer
classifier.add(Dense(units=6, activation='relu'))

# Adding output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the ann
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ann to the dataset
classifier.fit(X_train, Y_train, nb_epochs=100, batch_size=10)

# Predicting new data
Y_pred = classifier.predict(X_test)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)