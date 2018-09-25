
import numpy as np
import matplotlib.pyplot as plt

def plotBestFit(weights):
    '''
    画出决策边界
    '''
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def gradAscent(dataIn,classLables):
    '''
    梯度上升法
    '''
    # np.mat : 转换为矩阵类型
    dataMaxtrix = np.mat(dataIn)
    # 转置，用于多维数组，一二维的使用 np.T
    labelMat = np.mat(classLables).transpose()
    # 返回矩阵的形状
    m,n = np.shape(dataMaxtrix)
    alpha = 0.001
    maxCycles = 500 
    # 全1 矩阵
    weights = np.ones ((n,1))
    # 寻找500次
    for k in range (maxCycles):
        h = sigmoid(dataMaxtrix*weights)
        error = labelMat - h
        # 迭代算法，笔记里会有推倒的详细过程
        weights = weights + alpha * dataMaxtrix.transpose() * error
    return weights

def sigmoid(inX):
    '''
    sigmoid函数, Inx = w^T x, 
    '''
    return 1.0/(1+np.exp(-inX))

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('F:\\zhongjianwei\\pyscript\\机器学习实战\\Machine-learning-in-action-notes-and-code\\Logistic回归\\Ch05\\testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def main():
    dataArr,labelMat = loadDataSet()
    weights = gradAscent(dataArr,labelMat)
    plotBestFit(weights.getA()) #　降维

if __name__ == '__main__':
    main()