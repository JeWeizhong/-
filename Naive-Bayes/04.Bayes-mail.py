'''
示例：使用朴素贝叶斯对电子邮件进行分类
(1) 收集数据：提供文本文件。
(2) 准备数据：将文本文件解析成词条向量。
(3) 分析数据：检查词条确保解析的正确性。
(4) 训练算法：使用我们之前建立的trainNB0()函数。
(5) 测试算法：使用classifyNB()，并且构建一个新的测试函数来计算文档集的错误率。
(6) 使用算法：构建一个完整的程序对一组文档进行分类，将错分的文档输出到屏幕上。
'''
import re
import random
import numpy as np


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

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

def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('F:\\zhongjianwei\\pyscript\\机器学习实战\\Machine-learning-in-action-notes-and-code\\Naive-Bayes\\Ch04\\email\\email\\spam\\%d.txt' % i,encoding= 'ISO-8859-1').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('F:\\zhongjianwei\\pyscript\\机器学习实战\\Machine-learning-in-action-notes-and-code\\Naive-Bayes\\Ch04\\email\\email\\ham\\%d.txt' % i,encoding= 'ISO-8859-1').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = list(range(50)); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print ("classification error",docList[docIndex])
    print ('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText

def textParse(bigString):    #input is big string, #output is word list
    '''
    切分文本，大写转换成小写,去掉少于两个字母的字符串
    '''
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

def main():
    #emailText = open(r'F:\zhongjianwei\pyscript\机器学习实战\Machine-learning-in-action-notes-and-code\Naive-Bayes\Ch04\bayes.py','r').read()
    #print(textParse(emailText))
    spamTest()

if __name__ == '__main__':
    main()