'''
@author: xingyu, created on February 17, 2017, at Ecole Centrale de Lille
 https://github.com/xingyu-yan
# This programme is: Linear Regression
# reference: Coursera Machine Learning open course (Andrew Ng)
# reference: https://github.com/royshoo/mlsn
# edX online course Artificial Intelligence (AI) 
https://courses.edx.org/courses/course-v1:ColumbiaX+CSMM.101x+1T2017/info
'''
 
import numpy as np

def featureNormalize(X):
    mu = np.mean(X,axis=0)
    # numpy std is different from matlab std. The difference will get smaller when huge dataset is processed and the prediction will not be influenced
    sigma = np.std(X,axis=0)
    X_norm = np.divide(X-mu,sigma)
    return (X_norm,mu,sigma)

def computeCost(X,y,theta):
    #cost function: J = (1/2m)*sum((y_pred - y_real)**2)
    m = len(y)
    costJ = np.zeros(m)
    for k in range(m):
        X1 = (X[k,:])
        y1 = y[k]
        t = np.dot(X1,theta) - y1
        costJ[k] = t*t   
    totalCostJ = costJ.sum()/2/m    
    return (totalCostJ)

def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    temp0 = []
    temp1 = []
    for i in range(num_iters):
        for j in range(m):
            X1 = (X[j,:])
            y1 = y[j]
            t = np.dot(X1,theta) - y1
            temp0 = theta[0] - alpha/(len(X))*t*X[j,0]
            temp1 = theta[1] - alpha/(len(X))*t*X[j,1]
            theta[0] = temp0 
            theta[1] = temp1
            theta = np.hstack((theta[0],theta[1]))
        J_history[i] = (computeCost(X,y,theta))
    return (theta, J_history)
