# coding=utf-8
import math

def calcShannonEnt(dataSet):  
    #calculate the shannon value  
    numEntries = len(dataSet)  
    labelCounts = {}  
    for featVec in dataSet:      #create the dictionary for all of the data  
        currentLabel = featVec[-1]  
        if currentLabel not in labelCounts.keys():  
            labelCounts[currentLabel] = 0  
        labelCounts[currentLabel] += 1  
    shannonEnt = 0.0  
    for key in labelCounts:  
        prob = float(labelCounts[key])/numEntries  
        shannonEnt -= prob*math.log(prob,2) #get the log value  
    return shannonEnt  

def createDataSet():  
    # dataSet = [[1,"a",'yes'],  
    #            [1,"a", 'yes'],  
    #            [1,"b",'no'],  
    #            [0,"a",'no'],  
    #            [0,"a",'no']]  
    # labels = ['no surfacing','flippers']  
    dataSet=[
        ['man','child','low','no'],
        ['woman','child','low','no'],
        ['woman','child','low','no'],
        ['man','young','low','yes'],
        ['man','young','low','yes'],
        ['woman','young','low','yes'],
        ['woman','young','middle','no'],
        ['man','old','high','yes'],
        ['man','old','low','no'],
        ['woman','old','middle','no'],

        # ['woman','old','middle','no'],
        # ['woman','old','middle','no'],
        # ['woman','old','middle','no'],
        # ['woman','old','middle','no'],

        # ['man','old','middle','yes'],
        # ['man','old','middle','yes'],
        # ['man','old','middle','yes'],
        # ['man','old','middle','yes'],


        # ['man','low','no'],
        # ['woman','low','no'],
        # ['woman','low','no'],
        # ['man','low','yes'],
        # ['man','low','yes'],
        # ['woman','low','yes'],
        # ['woman','middle','no'],
        # ['man','high','yes'],
        # ['man','low','no'],
        # ['woman','middle','no'],
    ]

    # labels = ['gender','income']  
    labels = ['gender','age','income']  
    return dataSet, labels  

def splitDataSet(dataSet, axis, value):  
    retDataSet = []  
    for featVec in dataSet:  
        if featVec[axis] == value:      #abstract the fature  
            reducedFeatVec = featVec[:axis]  
            reducedFeatVec.extend(featVec[axis+1:])  
            retDataSet.append(reducedFeatVec)  
    return retDataSet  

def chooseBestFeatureToSplit(dataSet):  
    numFeatures = len(dataSet[0])-1  
    baseEntropy = calcShannonEnt(dataSet)  
    bestInfoGain = 0.0; bestFeature = -1  
    for i in range(numFeatures):  
        featList = [example[i] for example in dataSet]  
        uniqueVals = set(featList)  
        newEntropy = 0.0  
        for value in uniqueVals:  
            subDataSet = splitDataSet(dataSet, i , value)  
            prob = len(subDataSet)/float(len(dataSet))  
            newEntropy +=prob * calcShannonEnt(subDataSet)  
        infoGain = baseEntropy - newEntropy 
        # print infoGain 
        if(infoGain > bestInfoGain):  
            bestInfoGain = infoGain  
            bestFeature = i  
    # exit()
    return bestFeature  

def majorityCnt(classList):  
    # print classList
    # return 
    classCount = {}  
    for vote in classList:  
        if vote not in classCount.keys(): classCount[vote] = 0  
        classCount[vote] += 1  
    # print classCount
    tmp=[(k,classCount[k]) for k in classCount]
    tmp2=sorted(tmp, key=lambda row: row[1], reverse = True)
    # print tmp2
    # exit()
    # return vote
    # sortedClassCount = sorted(classCount.iteritems() as operator, key=operator.itemgetter(1), reverse=True)  
    return tmp2[0][0]  

def createTree(dataSet, labels):  
    # print dataSet
    classList = [example[-1] for example in dataSet]  
    # print len(dataSet[0])
    # return
    # the type is the same, so stop classify  
    if classList.count(classList[0]) == len(classList):  
        return classList[0]
    # print classList.count(classList[2])
    # return 
    # traversal all the features and choose the most frequent feature  
    # print len(dataSet[0])
    # return
    if (len(dataSet[0]) == 1):  
        return majorityCnt(classList)  
    bestFeat = chooseBestFeatureToSplit(dataSet)  
    # print bestFeat
    # return 
    bestFeatLabel = labels[bestFeat]  
    myTree = {bestFeatLabel:{}}  
    del(labels[bestFeat])  
    #get the list which attain the whole properties  
    featValues = [example[bestFeat] for example in dataSet]
    # print featValues
    # return ;  
    uniqueVals = set(featValues) 
    # print uniqueVals
    # return  
    for value in uniqueVals:  
        subLabels = labels[:]  
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  
    # print set(classList)
    maxclassnum=0
    maxclass=classList[0]
    for i in set(classList):
        if classList.count(i)>maxclassnum:
            maxclassnum=classList.count(i)
            maxclass=i
    myTree[bestFeatLabel]["default"]=maxclass
    return myTree  

myDat, labels = createDataSet() 
label=labels[:]
# print myDat
# print labels
myTree = createTree(myDat,label) 
# print  labels
print myTree 
# exit

def classify(inputTree, featLabels, testVec):  
    firstStr = inputTree.keys()[0]  
    # print firstStr
    # return 
    # print featLabels
    # return 
    secondDict = inputTree[firstStr]  
    featIndex = featLabels.index(firstStr)  
    # return 
    for key in secondDict.keys():  
        if testVec[featIndex] == key:  
            if type(secondDict[key]).__name__ == 'dict':  
                classLabel = classify(secondDict[key], featLabels, testVec)  
            else: classLabel = secondDict[key]  
    return classLabel 

# print classify(myTree,labels,[1,"a"]) 

# print classify(myTree,labels,['man','old','low']) 

tmp={'a':"123","b":{"c":1}}
print "a" 
# print tmp