import numpy as np
import operator

'''
对未知类别属性的数据集中的每个点依次执行以下操作：
(1) 计算已知类别数据集中的点与当前点之间的距离；
(2) 按照距离递增次序排序；
(3) 选取与当前点距离最小的k个点；
(4) 确定前k个点所在类别的出现频率；
(5) 返回前k个点出现频率最高的类别作为当前点的预测分类。
'''

def classify0(inX, dataSet, labels, k):
    # 返回行数
    dataSetSize = dataSet.shape[0]
    # 在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # 返回distances中元素从小到大排序后的索引值 
    sortedDistIndicies = distances.argsort()
    #print(sortedDistIndicies)  
    classCount={}          
    for i in range(k):
        #print(sortedDistIndicies[i]) # 1,0,3
        #print(labels[sortedDistIndicies[i]]) # A,A,B
        voteIlabel = labels[sortedDistIndicies[i]]  
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #python3中用items()替换python2中的iteritems()
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典        
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

if __name__ == '__main__':
    group, labels = createDataSet()
    print(group)
    result = classify0([0,0], group, labels, 3)
    print(result)