#Logistic Regression with a Neural Network mindset
# -*- coding:utf-8 -*-
"""
Created on Thu Jul 23 17:08:25 2020
@Author Mr.Lu
"""

import numpy as np
import h5py
from matplotlib import pyplot as plt
from lr_utils import load_dataset

num_iterations=2000 #迭代次数
learning_rate=0.005 #学习率
#tarain_set_y_orig为1x209的向量 标签
#test_set_y_orign为1x50的向量 标签

train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes=load_dataset() #加载数据

m_train=train_set_y_orig.shape[1] #训练集里面的图片数量
m_test=test_set_y_orig.shape[1] #测试集里面的图片数量
num_px=train_set_x_orig.shape[1] #图片尺寸

#将训练集的数据变成209x(64x64x3)的矩阵，然后进行转置
#将测试集的数据变成209x(64x64x3)的矩阵，然后进行转置
train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T  #将训练集数据降维并且转置，构成新的矩阵
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_set_x=train_set_x_flatten/255 #标准化数据
test_set_x=test_set_x_flatten/255

#定义激活函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

def initialize_with_zeros(dim):
    #dim为特征数量，即64*64*3个，创建一个(64*64*3)*1维的向量，不采用拓展b的形式
    w=np.zeros(shape=(dim,1))
    b=0  #偏差
    return (w,b)

def propagate(w,b,X,Y):
    m=X.shape[1] #样本数量  因为完成了转置，也就是说第二列开始为样本数量，第一列为特征数量
    #正向传播
    A=sigmoid(np.dot(w.T,X)+b)
    cost=(-1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))

    #反向传播
    dw=(1/m)*np.dot(X,(A-Y).T)
    db=(1/m)*np.sum(A-Y)

    grads={
            "dw":dw,
            "db":db
    }
    return (grads,cost)

#运行梯度下降来优化w和b
def optimize(w,b,X,Y):
    costs=[] #画代价曲线用

    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y)

        dw=grads["dw"]
        db=grads["db"]

        w=w-learning_rate*dw
        b=b-learning_rate*db
        if i%100==0:
            costs.append(cost) #保存每一次迭代完成后的代价
    params={
            "w":w,
            "b":b
    }
    grads={
            "dw":dw,
            "db":db
    }
    return (params,grads,costs)

def predict(w,b,X):
    m=X.shape[1]
    Y_prediction=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)

    A=sigmoid(np.dot(w.T,X)+b)

    for i in range(X.shape[1]):
        Y_prediction[0,i]=1 if A[0,i]>0.5 else 0

    return Y_prediction

def model(X_train,Y_train,X_test,Y_test):

    w,b=initialize_with_zeros(X_train.shape[0])
    parameters,grads,costs=optimize(w,b,X_train,Y_train)
    w=parameters["w"]
    b=parameters["b"]

    Y_prediction_test=predict(w,b,X_test)
    Y_prediction_train=predict(w,b,X_train)

    d={
        "costs":costs,
        "Y_prediction_test":Y_prediction_test,
        "Y_prediction_train":Y_prediction_train,
        "w":w,
        "b":b,
    }

    return d

d=model(train_set_x,train_set_y_orig,test_set_x,test_set_y_orig)

costs=np.squeeze(d["costs"])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iteration')
plt.title('Learning:'+str(learning_rate))
plt.show()
