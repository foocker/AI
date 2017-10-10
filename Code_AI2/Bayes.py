from numpy import *
'''
朴素贝叶斯,朴素二字在于简化模型,所增加的两个假设:1特征之间条件独立(降低了对数据量的需求),2每个特征同等重要.若将单词或者句子向量化,我们很容易利用朴素贝叶斯对文本进行分类.
本代码来源"机器学习实战"一书,增加了更详细的说明,以及少量改动.文本切片,向量化等都可以用py的现有库,比如nltk,jieba,tensorlayer.nlp...,所有后续由很多需要扩充...
'''
def loadDataSet():
    '''
    文本切片可换用jieba,tensorlayer等.
    '''
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    # 1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    '''
    将一篇(段)文本中的所有词(字)转化成一列表.
    '''
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)
    
def setofWords2Vec(vocabList, inputSet):
    '''
    构造句子向量.
    '''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:{0} is not in my Vocabulary!".format(word))
    return returnVec
    
def trainNB0(trainMatrix,trainCategory):
    '''
    计算在类别i情况下,句子(单词,字)出现的条件概率,以及类别自身概率.
    trainMatrix: 文本向量化矩阵
    trainCategory: 句子类别标签向量
    '''
    numTrainDocs = len(trainMatrix)    # 段落句子个数
    numWords = len(trainMatrix[0])     # 所有词组成的表的词个数
    pAbusive = sum(trainCategory)/float(numTrainDocs)    # 某类别概率
    p0Num = ones(numWords); p1Num = ones(numWords)       # change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                         # change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          # change to log()
    p0Vect = log(p0Num/p0Denom)          # change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    利用贝叶斯条件概率公式判断某句子属于i类的概率.分类器.
    vec2Classify: 要分类的向量
    p0Vec:　0类条件下,句子出现的概率向量(这里仅分两类)
    pClass1: 类别1的概率
    '''
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
def bagOfWords2VecMN(vocabList, inputSet):
    '''
    袋模型,考虑单词(字)的重复情况.
    '''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    '''
    测试模型
    '''
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setofWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setofWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage', 'make', 'love']
    thisDoc = array(setofWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))



