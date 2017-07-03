'''
@author: Xingyu YAN, created on May 20, 2017
# at University Lille1, France
# This program is for linear interpolation.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from LRfuncs import computeCost, gradientDescent 
# This two functions are from the linear regration 

# Loading data
print("Loading Data ...\n")
DataTrain = pd.read_csv('progra1.csv') 
A_col_train = np.matrix(DataTrain)
print(A_col_train.shape)
print(A_col_train[0,:])

newTable = np.reshape(A_col_train[:,15], (511, 15))
print(newTable.shape)
print(newTable[0,:])
    
X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# There are 15 year in total 
numberNotV = 0
for i in range(15):    
    newY = newTable[1,i] 
    if np.math.isnan(newY) == False:
        numberNotV += 1
newX = np.zeros(numberNotV) 
newY = np.zeros(numberNotV) 

tNotV = 0  
tV = 0
needForecost = np.zeros(15-numberNotV)  
   
for i in range(15):    
    new = newTable[1,i]
    print(new)     
    if np.math.isnan(new) == False:
        newY[tNotV] = new
        newX[tNotV] = i
        tNotV += 1
    else:
        needForecost[tV] = i
        tV += 1
print(newX.shape)
newXY = np.c_[newX,newY] 
print(newXY.shape) 
print(needForecost)     

# Part 1: Feature Normalization

#X = np.matrix(newX)
X = newX
y = newY

# Part 2: Gradient descent
#Some gradient descent settings
alpha = 0.01
num_iters = 100

a_ones = np.ones((len(X),1))
print(a_ones)

X = X.reshape(len(X),1)
print(X.shape)

X_new = np.hstack((a_ones,X))
print('print new X: ')

theta = np.zeros((2,1))
theta.shape = (len(theta),1)
print('theta looks like this ', theta)

X_theta = np.dot(X_new[len(X)-1,:],theta)
print('X multiply theta looks like this ', X_theta)

#calculate the cost by using the cost function 
J = computeCost(X_new,y,theta)
print('the final cost is: ', J)

(theta,J_history) = gradientDescent(X_new,y,theta,alpha,num_iters)
#(theta) = gradientDescent(X,y,theta,alpha,num_iters)
print ("Theta computed from gradient descent: ", theta)
print ("J_history are: ", J_history)

# Part 3: Visualizing 

#1 plot the Training data and the obtained line
plt.figure(1, figsize=(5, 3.75))
plt.plot(X,y,'ro')
plt.xlim(-1, 15)
plt.ylabel('y')
plt.xlabel('X')
plt.plot(X, np.dot(X_new,theta))

#Predict missing values for absent years
predict = np.zeros(tV)
for i in range(tV):
    yearN = int(needForecost[i])
    predict[i] = np.dot([1,yearN], theta)
    newTable[1,yearN] = predict[i]    
    plt.plot(yearN,predict[i],'bx', markersize=10)
print(newTable[1])
#plot the predictions
plt.show()

# data output in .csv file
import csv
b = open('Item_Zone.csv','a') 
a = csv.writer(b)
a.writerows(Item_Zone)  
b.close
print('Finished')
