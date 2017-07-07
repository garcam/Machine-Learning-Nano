# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 11:08:58 2017

@author: dcaramu
"""

#
# In this and the following exercises, you'll be adding train test splits to the data
# to see how it changes the performance of each classifier
#
# The code provided will load the Titanic dataset like you did in project 0, then train
# a decision tree (the method you used in your project) and a Bayesian classifier (as
# discussed in the introduction videos). You don't need to worry about how these work for
# now. 
#
# What you do need to do is import a train/test split, train the classifiers on the
# training data, and store the resulting accuracy scores in the dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')
# Limit to numeric data
X = X._get_numeric_data()
# Separate the labels
y = X['Survived']
# Remove labels from the inputs, and age due to missing data
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
from sklearn import datasets
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)


# The decision tree classifier
clf1 = DecisionTreeClassifier(random_state=0)
clf1.fit(X_train,y_train)
print "Decision Tree has accuracy: ",accuracy_score(y_test, clf1.predict(X_test))
print "Confusion matrix for this Decision Tree:\n",confusion_matrix(y_test,clf1.predict(X_test))
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(recall(y_test,clf1.predict(X_test)),precision(y_test,clf1.predict(X_test)))
print "Decision Tree F1 score: {:.2f}".format(f1_score(y_test, clf1.predict(X_test)))
# The naive Bayes classifier

clf2 = GaussianNB()
clf2.fit(X_train,y_train)
print "GaussianNB has accuracy: ",accuracy_score(y_test, clf2.predict(X_test))
print "GaussianNB confusion matrix:\n",confusion_matrix(y_test,clf2.predict(X_test))
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(recall(y_test,clf2.predict(X_test)),precision(y_test,clf2.predict(X_test)))
print "GaussianNB F1 score: {:.2f}".format(f1_score(y_test, clf2.predict(X_test)))

answer = { 
 "Naive Bayes Score": 0, 
 "Decision Tree Score": 0
}

confusions = {
 "Naive Bayes": 0,
 "Decision Tree": 0
}


########################################################
########################################################
########################################################

## Otro ejemplo pero ahora en lugar de un problema de clasificación vamos a ver uno de regresión


import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)

reg1 = DecisionTreeRegressor()
reg1.fit(X_train, y_train)
print "Decision Tree mean absolute error: {:.2f}".format(mae(y_test,reg1.predict(X_test)))
print "Decision Tree mean squared error: {:.2f}".format(mse(y_test, reg1.predict(X_test)))

reg2 = LinearRegression()
reg2.fit(X_train, y_train)
print "Linear regression mean absolute error: {:.2f}".format(mae(y_test,reg2.predict(X_test)))
print "Linear regression mean squared error: {:.2f}".format(mse(y_test, reg2.predict(X_test)))

results = {
 "Linear Regression": 0,
 "Decision Tree": 0
}

#######################################################
#######################################################
#######################################################

##Learning Curve
#Now that you have understood the Bias and Variance concepts let us learn about ways we can identify when our model performs well. The Learning Curve functionality from sklearn can help us in this respect. It allows us to study the behavior of our model with respect to the number of data points being considered to understand if our model is performing well or not.

#To start with , we have to import the module:

#from sklearn.learning_curve import learning_curve # sklearn 0.17
#from sklearn.model_selection import learning_curve # sklearn 0.18
#From the documentation, a reasonable implementation of the function would be as follows:

# learning_curve(
#        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)


# In this exercise we'll examine a learner which has high variance, and tries to learn
# nonexistant patterns in the data.
# Use the learning curve function from sklearn.learning_curve to plot learning curves
# of both training and testing error.
# CODE YOU HAVE TO TYPE IN IS IN LINE 35

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
# PLEASE NOTE:
# In sklearn 0.18, the import would be from sklearn.model_selection import learning_curve
from sklearn.learning_curve import learning_curve # sklearn version 0.17
from sklearn.cross_validation import KFold
from sklearn.metrics import explained_variance_score, make_scorer
import numpy as np

# Set the learning curve parameters; you'll need this for learning_curves
size = 1000
cv = KFold(size,shuffle=True)
score = make_scorer(explained_variance_score)

# Create a series of data that forces a learner to have high variance
X = np.round(np.reshape(np.random.normal(scale=5,size=2*size),(-1,2)),2)
y = np.array([[np.sin(x[0]+np.sin(x[1]))] for x in X])

def plot_curve():
    # Defining our regression algorithm
    reg = DecisionTreeRegressor()
    # Fit our model using X and y
    reg.fit(X,y)
    print "Regressor score: {:.4f}".format(reg.score(X,y))
    
    # TODO: Use learning_curve imported above to create learning curves for both the
    #       training data and testing data. You'll need reg, X, y, cv and score from above.
        
    train_sizes, train_scores, test_scores = learning_curve(
        reg, X, y, cv=cv)
    
    # Taking the mean of the test and training scores
    train_scores_mean = np.mean(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    
    # Plotting the training curves and the testing curves using train_scores_mean and test_scores_mean 
    plt.plot(train_sizes ,train_scores_mean,'-o',color='b',label="train_scores_mean")
    plt.plot(train_sizes,test_scores_mean ,'-o',color='r',label="test_scores_mean")
    
    # Plot aesthetics
    plt.ylim(-0.1, 1.1)
    plt.ylabel("Curve Score")
    plt.xlabel("Training Points")
    plt.legend(bbox_to_anchor=(1.1, 1.1))
    plt.show()

plot_curve()