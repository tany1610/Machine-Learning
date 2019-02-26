# Natural Language Processing

# Importing the libraires
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
sp = PorterStemmer()
reviews_list = []
for i in range(0, 1000):
    review = dataset['Review'][i]
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [sp.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    reviews_list.append(review)
    
# Creating the Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(reviews_list).toarray()
Y = dataset.iloc[:, 1].values

# Split the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=0)

# Fitting the Classifier to the dataset
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, Y_train)

# Predicting new data
Y_pred = classifier.predict(X_test)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)