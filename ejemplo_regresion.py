#!/usr/bin/python

def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module, create, and train your regression
    ### name your regression reg
    
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit(ages_train, net_worths_train)
    
    ### your code goes here!
    
    
    
    return reg

import numpy
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from studentRegression import studentReg
from class_vis import prettyPicture, output_image

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()


reg = studentReg(ages_train, net_worths_train)


plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2)
plt.xlabel("ages")
plt.ylabel("net worths")


plt.savefig("test.png")
output_image("test.png", "png", open("test.png", "rb").read())

print "katie's net worth prediction: ", reg.predict([27])
print "slope: ", reg.coef_
print "intercept: ", reg.intercept_

print "\n #######stats on test dataset #########\n"
print "r-squared score: ", reg.score(ages_test, net_worths_test)

print "\n #######stats on training dataset #########\n"
print "r-squared score: ", reg.score(ages_train, net_worths_train)

##otra manera de hacer scatter plots
plt.scatter(ages, net_worths)
plt.plot(ages, reg.predict(ages), color='blue', linewidth=3)
plt.xlabel("ages")
plt.ylabel("net worths")
plt.show()


