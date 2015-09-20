#coding=utf-8
import sys
from numpy import *

#返回分词后的文章单词列表，以及类别；
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    #1代表侮辱性文字，0代表正常语言
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

#把输入的dataSet里面的单词去重，得到一个词表，输出一个list
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        #|操作符的作用是求并集
        vocabSet = vocabSet | set(document)
    
    return list(vocabSet)

#参数vacabList表示去重后的此表list，inputSet表示文档输入文档的单词；
#把输入文档表示成向量，向量的长度是去重后list的长度；如果inputSet中的单词在vocabList中，那么对应向量中的值为1；
#返回向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "The word: %s is not in my vocabulary!" % word
    
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    #训练的文档数量；
    numTrainDocs = len(trainMatrix)
    #文档的向量长度；
    numWords = len(trainMatrix[0])
    #侮辱性留言所占比例
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #初始化概率
    #为了防止其中一个概率为0，导致乘积为0，需要把所有词出现的个数设置为1，并将分母设置为2；
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    #单词出现的次数比上所有单词出现的次数；
    #为了防止下溢出（这是由于太多小数相乘造成的），可以堆乘积取对数，可以避免下溢出或者浮点数舍入导致的错误；
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
def testingNB():
    listOfPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classifed as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classifed as: ', classifyNB(thisDoc, p0V, p1V, pAb)

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []; classList = []; fullText = []
    #加载数据
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        #一行docList表示一篇文档
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = range(50); testSet=[]
    #随机选择10篇文档作为测试
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        #选中测试的不再用作训练
        del(trainingSet[randIndex])

    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0v, p1v, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0

    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testSet)

if __name__ == '__main__':
    '''
    #返回分词后的文章单词列表，以及类别；
    listOfPosts, listClasses = loadDataSet()
    #把输入的dataSet里面的单词去重，得到一个词表，输出一个list
    myVocabList = createVocabList(listOfPosts)
    #returnVec = setOfWords2Vec(myVocabList, listOfPosts[0])

    #参数vacabList表示去重后的此表list，inputSet表示文档输入文档的单词；
    #把输入文档表示成向量，向量的长度是去重后list的长度；如果inputSet中的单词在vocabList中，那么对应向量中的值为1；
    #返回向量
    trainMat = []
    #把输入的文章转化为向量；
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print pAb
    print p0V
    print p1V
    '''
    #testingNB()
    spamTest()
