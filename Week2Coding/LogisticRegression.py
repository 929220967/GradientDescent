# -*- coding:utf-8 -*-
"""
  Create on
  @Author:Mr.Lu
"""
import numpy as np
import h5py
from matplotlib import pyplot as plt
from utils import load_dataset

IteratorNum=4000  #迭代次数
LearningRate=0.05 #学习率

#目前train_set_x是209,64,64,3维矩阵
#目前train_set_y是1*209的矩阵
#目前test_set_x是50,64,64,3的矩阵
#目前test_set_y是1*50的矩阵
train_set_x,train_set_y,test_set_x,test_set_y,classes=load_dataset()

#降维并转置
train_set_x_flatten=train_set_x.reshape(train_set_x.shape[0],-1).T  #12288*209
test_set_x_flatten=test_set_x.reshape(test_set_x.shape[0],-1).T   #12288*50

train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255

def sigmoid(z):
    return 1/(1+np.exp(-z))

def initilaize_with_zeros(colLength):

    w=np.zeros((colLength,1))
    b=0
    return w,b

def Gradient_descent(X,Y,w,b):
    m=X.shape[1]
    #向前传播
    A=sigmoid(np.dot(w.T,X)+b)
    cost=(-1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    #X此时是12288*209的矩阵,Y是1*209的矩阵向量，A-Y是？*209的矩阵
    #反向传播
    dw=(1/m)*np.dot(X,(A-Y).T)
    db=(1/m)*np.sum(A-Y)

    return dw,db,cost

def optimaize(X,Y,w,b):

    costs=[]
    for i in range(IteratorNum):
        dw,db,cost=Gradient_descent(X,Y,w,b)

        w=w-LearningRate*dw
        b=b-LearningRate*db
        if i%100==0:
            costs.append(cost)

    return w,b,costs

def predict(X,w,b): #此时的X为测试集 12288*50的矩阵
    m=X.shape[1]
    Y_prediction=np.zeros((1,m))  #定义1*50的向量
  #  w=w.reshape(X.shape[0],1)   #将w变为12288*1的向量

    A=sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):
        Y_prediction[:i]=1 if A[0:i]>0.5 else 0

    return Y_prediction

def model(train_set_x,train_set_y,test_set_x,test_set_y):
    w,b=initilaize_with_zeros(train_set_x.shape[0])
    w,b,costs=optimaize(train_set_x,train_set_y,test_set_x,test_set_y)

    Y_prediction_test=predict(test_set_x,w,b)

    return w,b,costs

w,b,costs=model(train_set_x,train_set_y,test_set_x,test_set_y)

costs=np.squeeze(costs)
plt.plot(costs)
plt.xlabel('iterations:')
plt.ylabel('Cost:')
plt.title('LearningRate:')
plt.show()