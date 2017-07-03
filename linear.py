'''
@author: Xingyu YAN, created on May 20, 2017
# at University Lille1, France
# This program is for linear interpolation.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stage.LRfuncs import computeCost, gradientDescent

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

numberNotV = 0
for i in range(15):    
    newY = newTable[1,i] 
    if np.math.isnan(newY) == False:
        numberNotV += 1
newX = np.zeros(numberNotV) 
newY = np.zeros(numberNotV) 
print(newX) 

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
print(newX)
print(newY)  
newXY = np.c_[newX,newY] 
print(newXY.shape) 
print(newXY)   
print(needForecost)     

# Part 1: Feature Normalization

#X = np.matrix(newX)
X = newX
print(X)
print(len(X))

y = newY

# Part 2: Gradient descent
#Some gradient descent settings
alpha = 0.01
num_iters = 100

a_ones = np.ones((len(X),1))
print(a_ones)

X = X.reshape(len(X),1)
print(X.shape)
#print(X)

X_new = np.hstack((a_ones,X))
print('print new X: ')
#print(X_new)
print(X_new.shape)
theta = np.zeros((2,1))
theta.shape = (len(theta),1)
print('theta looks like this ')
print(theta)
X_theta = np.dot(X_new[len(X)-1,:],theta)
print('X multiply theta looks like this ')
print(X_theta)

#calculate the cost by using the cost function 
J = computeCost(X_new,y,theta)
print('the final cost is: ')
print(J)

(theta,J_history) = gradientDescent(X_new,y,theta,alpha,num_iters)
#(theta) = gradientDescent(X,y,theta,alpha,num_iters)
print ("Theta computed from gradient descent:")
print (theta)
'''print ("J_history are:")
print (J_history)'''

# Part 3: Visualizing 

#1 plot the Training data and the obtained line
plt.figure(1, figsize=(5, 3.75))
plt.plot(X,y,'ro')
plt.xlim(-1, 15)
plt.ylabel('y')
plt.xlabel('X')
plt.plot(X, np.dot(X_new,theta))
#plt.legend('Training data', 'Linear regression')

#Predict missing values for absent years
predict = np.zeros(tV)
for i in range(tV):
    yearN = int(needForecost[i])
    predict[i] = np.dot([1,yearN], theta)
    newTable[1,yearN] = predict[i]    
    plt.plot(yearN,predict[i],'bx', markersize=10)
print(newTable[1])
#plot the predictions
'''plt.plot(0,predict[0],'bx', markersize=10)
plt.plot(3,predict[2],'bx', markersize=10)'''
plt.show()

'''
import csv
b = open('Item_Zone.csv','a') 
a = csv.writer(b)
a.writerows(Item_Zone)  
b.close
print('Finished')'''