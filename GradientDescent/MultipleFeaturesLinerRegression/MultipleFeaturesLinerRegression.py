##多特征线性回归的房价预测
import numpy as  np
from  matplotlib import pyplot as plt

np.set_printoptions(suppress=True)  #禁止科学计数法
plt.rcParams['font.sans-serif'] = ['SimHei'] #允许画图中中文出现
plt.rcParams['axes.unicode_minus']=False  #解决画图中出现负数刻度显示异常的情况
itersNum=1000 #迭代次数
learnRate=0.01 #学习率

#1、首先读取文件中的数据
def loadFile(path):
    return np.loadtxt(path,dtype=np.float64,delimiter=',')

#2、定义一个线性回归函数
def linerRegression():
    data=loadFile('data.csv') #读取文件数据
    x_data=np.array(data[:,0:-1])
    y_data=np.array(data[:,-1]).reshape(-1,1)

    x_data=meanNormalization(x_data)
    plotMeanNormalization(x_data)

    x_data=np.hstack((np.ones((len(y_data),1)),x_data))  #插入一列为1的数组

    colNmus=x_data.shape[1]  #计算出行数，以便确定所求参数个数
    theta=np.zeros((colNmus,1))  #构建一个参数向量

    theta,costAll=gradientDescent(x_data,y_data,theta)

    plotCostCurve(costAll)
    plotLinearRegression(x_data,theta,y_data)
    return theta

#3、均值归一化函数
def meanNormalization(x_data):
    columnsMean=np.mean(x_data,0)  #求出每一列的均值，0表示求列的均值，1表示求行的均值
    columnsStd=np.std(x_data,0)    #求出每一类的标准差，0表示求列的标准差，1表示求行的标准差

    for i in range(x_data.shape[1]):  #归一化   每一列的中的值减去均值，然后除去标准差  shape[0]输出行数，shape[1]输出列数
        x_data[:,i]=(x_data[:,i]-columnsMean[i])/columnsStd[i]

    return x_data

#4、显示均值归一化的效果，也就是散点图
def plotMeanNormalization(x_data):
    plt.scatter(x_data[:,0],x_data[:,1])
    plt.title('数据均值归一化效果')
    plt.savefig('均值归一化效果.png') #保存拟合图片
    plt.show()

#5、核心算法，开始进行迭代，进行梯度下降
def gradientDescent(x_data,y_data,theta):
    theta_num=len(theta)
    theta_temp=np.matrix(np.zeros((theta_num,itersNum)))  #为了同步更新权重用，保存每一次迭代的结果

    costAll=np.zeros((itersNum,1))  #保存代价

    for i in range(itersNum):
        hypothesis=np.dot(x_data,theta)
        theta_temp[:,i]=theta-(learnRate/len(y_data))*(np.dot(np.transpose(x_data),hypothesis-y_data))
        theta=theta_temp[:,i]
        costAll[i]=costFunction(x_data,y_data,theta)

    return theta,costAll

#6、计算代价函数
def costFunction(x_data,y_data,theta):
    return np.sum(np.power(np.dot(x_data,theta)-y_data,2))/(2*len(y_data))

#7、为了检验算法能否正确执行，现在将代价以图像的形式展现出来
def plotCostCurve(costAll):
    x=np.arange(0,itersNum)
    plt.plot(x,costAll)
    plt.xlabel('迭代次数')
    plt.ylabel('代价值')
    plt.title('代价随迭代次数变化曲线')
    plt.savefig('CostCurve.png')
    plt.show()

#8、将拟合的过程以3D立体图像形式展现出来
def plotLinearRegression(x_data,theta,y_data):
    plt.figure(figsize=(8,10))
    x=x_data[:,1]
    y=x_data[:,2]

    theta=theta.flatten()

    z=theta[0,0]+(theta[0,1]*x)+(theta[0,2]*y)

    ax=plt.subplot(211,projection='3d')
    ax.plot_trisurf(x,y,z)
    ax.scatter(x_data[:,1],x_data[:,2],y_data,label='实际数据')

    ax.set_xlabel('房屋大小')
    ax.set_ylabel('房间数')
    ax.set_zlabel('价格')
    plt.savefig('3d拟合theta值.png')
    plt.show()


print(linerRegression())  #执行算法

