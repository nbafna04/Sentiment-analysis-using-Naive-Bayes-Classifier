import string
import math
import random
import statistics
import matplotlib.pyplot as plt
import numpy as np
import sys

def read_text_file(filename):
    file = open(filename, "r")
    dict = {}
    for line in file:
        x = line.split("\t")
        if x[1].rstrip() in dict:
            dict[x[1].rstrip()].append(x[0])
        else:
            dict[x[1].rstrip()] = [x[0]]
    return (dict)

def splitDataForCrossValidation(dict, k):

    #shuffle data for in document dictionary
    dict['0'] = random.sample(dict['0'], len(dict['0']))
    dict['1'] = random.sample(dict['1'], len(dict['1']))

    TotalNoOfRecordsFoldwise = round(len(dict['0']) + len(dict['1'])) / k

    # since there are 2 clases number of records in each class will be
    numOfDocFoldwiseEachClass = int(TotalNoOfRecordsFoldwise/2)

    def divide_list_to_chunks(my_list, n):
        return [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n )]

    kFoldDataDict = {}
    for cls in dict:
        kFoldDataDict[cls] = divide_list_to_chunks(dict[cls], numOfDocFoldwiseEachClass)

    # kFoldDataDict has shuffled records of 0 and 1 with 10 list each of 50 docs
    return kFoldDataDict

def calculateAccuracy(kFoldDataDict,m):

    def createTrainingSet(myList, index):
        myList = myList[:index] + myList[index+1 :]
        return [j for i in myList for j in i]

    subSamples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.8 ,0.9, 1]
    LearningCurveAvg = {}
    LearningCurveStd = {}
    n = len(kFoldDataDict['0'])
    testData = {}
    trainingData = {}

    for subsample in subSamples:
        i = 0
        tempForAvg = []
        while i < n:

            testData['0'] = kFoldDataDict['0'][i]
            testData['1'] = kFoldDataDict['1'][i]
            trainingData['0'] = createTrainingSet(kFoldDataDict['0'], i)
            subSampleIdx = int(len(trainingData['0']) * subsample)
            trainingData['0'] = trainingData['0'][:subSampleIdx]

            trainingData['1'] = createTrainingSet(kFoldDataDict['1'], i)
            subSampleIdx1 = int(len(trainingData['1']) * subsample)
            trainingData['1'] = trainingData['1'][:subSampleIdx1]

            # passing m value as 1
            myList = MAP(buildDataset(trainingData), m)
            totalCorrectPredict = 0
            total = 0

            for cls in testData:
                for sentence in testData[cls]:
                    if predict(sentence, myList[0], myList[1], myList[2]) == int(cls):
                        totalCorrectPredict += 1
                    total += 1

            accuracy = (totalCorrectPredict/total) * 100
            tempForAvg.append(accuracy)
            i += 1
        LearningCurveStd[subsample] = statistics.stdev(tempForAvg)
        LearningCurveAvg[subsample] = Average(tempForAvg)

    return [LearningCurveAvg, LearningCurveStd]

def calculateAccuracyForSmoothing(kFoldDataDict,m):
    def createTrainingSet(myList, index):
        myList = myList[:index] + myList[index+1 :]
        return [j for i in myList for j in i]

    n = len(kFoldDataDict['0'])
    testData = {}
    trainingData = {}
    i = 0
    tempForAvg = []
    while i < n:

        testData['0'] = kFoldDataDict['0'][i]
        testData['1'] = kFoldDataDict['1'][i]
        trainingData['0'] = createTrainingSet(kFoldDataDict['0'], i)
        trainingData['1'] = createTrainingSet(kFoldDataDict['1'], i)
        # passing m value as 1
        myList = MAP(buildDataset(trainingData), m)
        totalCorrectPredict = 0
        total = 0

        for cls in testData:
            for sentence in testData[cls]:
                if predict(sentence, myList[0], myList[1], myList[2]) == int(cls):
                    totalCorrectPredict += 1
                total += 1

        accuracy = (totalCorrectPredict/total) * 100
        tempForAvg.append(accuracy)
        i += 1
    Avg = Average(tempForAvg)
    Std = statistics.stdev(tempForAvg)

    return [Avg, Std]

# removing punctuations
def preprocessDatafile(sentence):
    return sentence.lower().translate(str.maketrans('', '', string.punctuation))

def buildDataset(documents):

    distinctWords = {'0':[],'1':[]}
    frequencyWords = {'0':{},'1':{}}

    for cls in documents:
        for sentence in documents[cls]:
            # removing punctuation and making each word as lowercase
            sentence = preprocessDatafile(sentence)
            for eachWord in sentence.split(" "):
                if eachWord not in distinctWords[cls]:
                    distinctWords[cls].append(eachWord)
                if eachWord not in frequencyWords[cls]:
                    frequencyWords[cls][eachWord] = 1
                else:
                    frequencyWords[cls][eachWord] += 1

    return [frequencyWords, distinctWords, documents]

def MAP(parsedData,m):
    frequencyWords = parsedData[0]
    distinctWords = parsedData[1]
    documents = parsedData[2]

    totalPositiveWords = 0
    totalNegativeWords = 0

    for cls in frequencyWords:
        for each in frequencyWords[cls]:
            if cls == '0':
                totalNegativeWords += frequencyWords[cls][each]
            else:
                totalPositiveWords += frequencyWords[cls][each]

    newdist = distinctWords['0'] + distinctWords['1']
    countOfVocab = len(set(newdist))
    totalNoPositiveDoc = len(documents['1'])
    totalNoNegativeDoc = len(documents['0'])
    totalNoDoc = totalNoNegativeDoc + totalNoNegativeDoc
    Prior_Positive = totalNoPositiveDoc/totalNoDoc
    Prior_Negative = totalNoNegativeDoc/totalNoDoc

    mapToken = {'0':{}, '1':{}}
    newdist = distinctWords['0'] + distinctWords['1']

    for cls in ['0','1']:
        for eachWord in set(newdist):
            if eachWord in frequencyWords[cls]:
                if cls == '0':
                    mapToken[cls][eachWord] = (((frequencyWords[cls][eachWord]) + m) / float(totalNegativeWords + (m*countOfVocab)))
                else:
                    mapToken[cls][eachWord] = (((frequencyWords[cls][eachWord]) + m) / float(totalPositiveWords + (m*countOfVocab)))
            else:
                if cls == '0':
                    mapToken[cls][eachWord] = ((0 + m) / (totalNegativeWords + float(m*countOfVocab)))
                else:
                    mapToken[cls][eachWord] = ((0 + m) / (totalPositiveWords + float(m*countOfVocab)))
    return [mapToken, Prior_Positive, Prior_Negative]

def getPositiveWordProb(word, mapToken):
    if word in mapToken['1']:
        return (mapToken['1'][word])
    else:
        return None

def getNegativeWordProb(word, mapToken):
    if word in mapToken['0']:
        return (mapToken['0'][word])
    else:
        return None


def predict(sentence, mapToken, Prior_Negative, Prior_Positive):

    #pre process sentence
    sentence = preprocessDatafile(sentence)
    wordPositiveProdList = []
    wordNegatveProdList = []

    for eachWord in sentence.split():
        #check the probability in mapToken
        wordPositiveProdList.append(getPositiveWordProb(eachWord, mapToken))
        wordNegatveProdList.append(getNegativeWordProb(eachWord, mapToken))

    #calculate positive conditional probablity
    positiveProb = 1
    negativeProb = 1
    for prob in wordPositiveProdList:
        if prob:
            positiveProb *= (prob)

    # add log of positive prior to positive probability
    positiveCondProb = (Prior_Positive) * positiveProb

    for prob in wordNegatveProdList:
        if prob:
            negativeProb *= (prob)

    # add log of negative prior to negative probability
    negativeCondProb = (Prior_Negative) * negativeProb

    return 1 if positiveCondProb > negativeCondProb else 0

def Average(lst):
    return sum(lst) / len(lst)

def plotLearningCurve(smooth1, smooth2, filename):
    fig, ax = plt.subplots()
    # Define labels, positions, bar heights and error bar heights
    labels = ['0.1N','0.2N','0.3N','0.4N','0.5N','0.6N','0.7N','0.8N','0.9N','N']
    x_pos = np.arange(len(labels))

    x_dict1 = smooth1[0]
    x1 =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.8 ,0.9, 1]
    y_dict1 = smooth1[0]
    y1 = (y_dict1.values())
    yerr1 = smooth1[1].values()

    y_dict2 = smooth2[0]
    y2 = (y_dict2.values())
    yerr2 = smooth2[1].values()

    ax.errorbar(x1, y1,
                yerr=yerr1,
                label= "m=0")

    ax.errorbar(x1, y2,
                yerr=yerr2,
                label= "m=1")

    ax.legend(loc='upper left')

    ax.set_xlabel('Train Set Size')
    ax.set_ylabel('Accuracy per Size')
    plt.xticks(np.arange(min(x1), max(x1) + 0.1, 0.1))
    plt.yticks(np.arange(min(min(y1),min(y2)), max(max(y1), max(y2)), 5))
    ax.set_title(
        'Average accuracies and standard deviations as a function of smoothing parameter with 10 folds'
        ' for file ' + filename)
    plt.show()


def plotSmoothing(doc,filename):
    m = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    smoothingDictAvg = {}
    smoothingDictStd = {}
    for each in m:
        avgAndStd = calculateAccuracyForSmoothing(splitDataForCrossValidation(doc, 10), each)
        smoothingDictAvg[each] = avgAndStd[0]
        smoothingDictStd[each] = avgAndStd[1]
    fig, ax = plt.subplots()
    x1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y1 = smoothingDictAvg.values()
    yerr1 = smoothingDictStd.values()
    ax.errorbar(x1, y1,
                yerr=yerr1, fmt='-o')

    ax.legend(loc='upper left')
    ax.set_xlabel('smoothing factor')
    ax.set_ylabel('Average Accuracy')
    plt.xticks(np.arange(0, 10+1))
    plt.yticks(np.arange(40, max(y1), 10))
    # plt.yticks()
    ax.set_title(
        'Averages of the accuracy and standard deviations (as error bars) '
        'as a function of train set size for file ' + filename)
    plt.show()

# Main Function
if __name__ == "__main__":
    filename = sys.argv[1]
    doc = read_text_file(filename)
    print("Running Experiment 1")
    smooth1 = calculateAccuracy(splitDataForCrossValidation(doc, 10), 0)
    smooth2 = calculateAccuracy(splitDataForCrossValidation(doc, 10), 1)
    plotLearningCurve(smooth1, smooth2, filename)
    print("Running Experiment 2")
    plotSmoothing(doc, filename)