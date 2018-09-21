'''
Created on Oct 19, 2010

@author: Peter
'''
import numpy as np
import random

def loadDataSet():
    '''
    导入数据
    '''
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1是侮辱性词语，0不是
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
                 
def createVocabList(dataSet):
    '''
    计算单词出现的次数，利用的是集合中元素不重复的特性
    '''
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def bagOfWords2VecMN(vocabList, inputSet):
    '''
    与setOfWords2Vec差不多
    '''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 #这里是对的出现的单词计数
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    '''

    '''
    numTrainDocs = len(trainMatrix) #有多少行
    numWords = len(trainMatrix[0])  # 第一行有多少个单词
    pAbusive = sum(trainCategory)/float(numTrainDocs) # 这里看不懂
    # 构造全1矩阵
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)      #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1: #
            p1Num += trainMatrix[i] # 矩阵相加
            p1Denom += sum(trainMatrix[i]) # 矩阵的和
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        #print(trainMatrix[i])    
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def main():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    # 统计每句话单词是否在集合中，1代表出现，0代表没有
    for postinDoc in listOPosts:
        trainMat.append(bagOfWords2VecMN(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(trainMat,listClasses)
    print(p0V,p1V,pAb)

if __name__ == '__main__':
    main()
