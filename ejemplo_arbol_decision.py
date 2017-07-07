def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer

    from sklearn.tree import DecisionTreeClassifier
    cl = DecisionTreeClassifier(min_samples_split=50) #con la opcion min_sampl
    #es_split no particiona un nodo si no alcanza ese minimo de observaciones
    clf = cl.fit(features_train, labels_train)      
    
    return clf


import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()



### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)




#### grader code, do not modify below this line

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())


### be sure to compute the accuracy on the test set
from sklearn.tree import DecisionTreeClassifier
cl2 = DecisionTreeClassifier(min_samples_split=2)
cl50 = DecisionTreeClassifier(min_samples_split=50)
clf2 = cl2.fit(features_train, labels_train)
clf50 = cl50.fit(features_train, labels_train)
acc_min_samples_split_2 = clf2.score(features_test, labels_test)
acc_min_samples_split_50 = clf50.score(features_test, labels_test)
def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}
  
import math
-0.5*math.log(0.5,2)-0.5*math.log(0.5,2)
