#单变量房价预测-线性回归

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
data=np.genfromtxt('data.csv',delimiter=',')
x_data=data[:,0]
y_data=data[:,1]
k=0  #设置变量权重
lr=0.0001  #学习率
epochs=150  #迭代次数
b=0   #偏移量

def loadFile(path):
    return np.genfromtxt(path,delimiter=',')

def costFunction(x_data,y_data):  #定义的代价函数
    sum=0
    for i in range(len(x_data)):
        sum+=(x_data[i]*k+b-y_data[i])**2
    return sum/float(len(x_data))/2

def gradient_descent_runner(x_data,y_data,b,k,lr,epochs):
    for j in range(epochs):
        temp_k=0
        temp_b=0
        for i in range(len(x_data)):
           temp_k+=(k*x_data[i]+b-y_data[i])*x_data[i]
           temp_b+=k*x_data[i]+b-y_data[i]
        b-=lr*temp_b/len(x_data)
        k-=lr*temp_k/len(x_data)
    return  b,k

b,k=gradient_descent_runner(x_data,y_data,b,k,lr,epochs)
#plt.tick_params(labelsize=10)  #设置刻度
plt.title('单变量房价预测-线性回归')
plt.xlabel('面积:平米')
plt.ylabel('价格:万元')
plt.plot(x_data, y_data, 'b.')  #绘制散点图
plt.plot(x_data, k * x_data + b, 'r')  #绘制拟合回归直线
plt.savefig('PredictPrice.png')  #保存生成的图片
plt.show()