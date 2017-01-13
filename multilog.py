#!/usr/bin/env python

#imports
import math
from scipy.optimize import fmin_bfgs
from numpy import array
import random
import numpy

#global constants
numOfFeatures = 16
iterations = 135
eps = 1.0e-3

#main function
def main():
    features,label, classes = processData("trainingSet.tra", "trainSet")
    iterateLabel = uniqueLabels(label)
    labelLength = len(iterateLabel)
    featureLength = len(features[0])
    
    #initialize weights to 0.0
    weights = [0.0] * 16
    
    calculateNegLogLikelihood(weights, features, label, labelLength) #calculate negative loglikelihood
    calculateGradDescent(weights, features, label, labelLength)      #calculate gradients
    weights = array([0.0] * (labelLength * featureLength))            
    gradients = multinomialLogistic(weights, features, label)        #minimize gradients
    
    #updated gradients
    testingSetBetas = gradients[0]                 
    
    #predict on testing Set
    testFeatures, testLabel, testClasses = processData("testingSet.txt", "testSet")
    iterateTestLabel = uniqueLabels(testLabel)
    labelLengthTest = len(iterateTestLabel)
    testingClassification(testingSetBetas, testFeatures, testLabel, labelLengthTest)

#process the given input file 
def processData(filename, typeOfSet):
    data = open(filename, "r").read()
    data = data.replace(',', " ")
    data = data.strip().split("\n")
    records = []
    for i, line in enumerate(data):
        records.append(line)
    
    #shuffle input records if training Set    
    #if typeOfSet == "trainSet":
     #   random.shuffle(records)
    
    features = []
    label = []
    classes = None
    notClassified = True
    intercept = True
    for rec in records:
        cols = rec.split()
        if notClassified: 
            notClassified = False
            classes = cols
            continue
        label.append(int(cols[len(cols) - 1]))
        singleRecord = [float(c) for c in cols[0:len(cols) - 2]]
        if intercept:
            singleRecord.insert(0, 1.0)
        features.append(singleRecord)
    
    #calculate z-scores
    featureMean = calculateMean(features)
    featureSd = calculateSd(features, featureMean)
    ZScores = calculateZScores(features, featureMean, featureSd)
    
    return ZScores, label, classes

#get iterable labels
def uniqueLabels(label):
    uniqueLabelSet = {}
    for l in label:
        if l in uniqueLabelSet:
            uniqueLabelSet[l] += 1
        else:
            uniqueLabelSet[l] = 0
    return (uniqueLabelSet)

#calculate z-scores
def calculateZScores(dataSet, meanVector, sdVector):
    zMap = [0.0] * len(dataSet)
    length = len(dataSet)
    for rec in range(len(dataSet)):
        zScoreRecord = [0.0] * numOfFeatures
        record = dataSet[rec]
        for i in range(len(record)):
            x = record[i]
            mean = meanVector[i]
            sd = sdVector[i]
            if sd == 0.0:
                zScore = float(x)
            else:
                zScore = (float(x) - float(mean)) / float(sd)
                zScoreRecord[i] = zScore
            
        zMap[rec] = zScoreRecord
    
    return zMap
 
#calculate standard deviation
def calculateSd(dataSet, meanVector):
    sdVector = [0.0] * numOfFeatures
    diffVector = [0.0] * numOfFeatures
    length = len(dataSet)
    for i in range(numOfFeatures):
        for rec in dataSet:
            diffVector[i] += math.pow((rec[i] - meanVector[i]), 2)
            
        sdVector[i] = math.sqrt(float(diffVector[i]) / length)
    
    return sdVector
 
#calculate mean
def calculateMean(dataSet):
    meanVector = [0.0] * numOfFeatures
    sumVector = [0.0] * numOfFeatures
    length = len(dataSet)
    for i in range(numOfFeatures):
        for rec in dataSet:
            sumVector[i] += rec[i]
        
        meanVector[i] = float(sumVector[i]) / length
    
    return meanVector
    
#minimize gradients using bfgs algorithm
def multinomialLogistic(weights, features, label):
    labelLength = len(uniqueLabels(label))
    def fx(wts):
        return calculateNegLogLikelihood(wts, features, label, labelLength)
    def gx(wts):
        return calculateGradDescent(wts, features, label, labelLength)
    
    #using the scipy's fmin_bfgs algorithm
    minGradients = fmin_bfgs(fx , weights, fprime=gx, epsilon = eps, maxiter=iterations,  full_output = True, disp=False, retall=False)
    return minGradients

#calculate negative loglikelihood
def calculateNegLogLikelihood(weights, features, label, labelLength):
    likelihood = 0
    for row, col in zip(features,label):
        hw = [0.0] * labelLength
        sumWx = 0.0
        for i in range(labelLength):
            x = i * numOfFeatures
            wx = sum([(a * b) for (a,b) in zip(row, weights[x: x + numOfFeatures])])
            hw[i] = float(wx)
            sumWx += math.exp(wx)
        likelihoodDiff = hw[col] - math.log(float(sumWx))
        likelihood += likelihoodDiff        
    return -float(likelihood)
 
#calculate probabilities per class for a given record
def calculateProbabilities(row, weights, labelLength):
    hw = [0.0] * labelLength
    sumWx = 0.0
    for i in range(labelLength):
        x = i * numOfFeatures
        wx = sum([(a * b) for (a,b) in zip(row, weights[x: x + numOfFeatures])])
        expWx = math.exp(wx)
        hw[i] = float(expWx)
        sumWx += expWx
               
    probPredictions = [float(hw[j]) /float(sumWx) for j in range(labelLength)]
    return probPredictions
 
#calculate gradient descent
def calculateGradDescent(weights, features, label,  labelLength):  
    likelihood = 0
    gradients = array([0.0] * (labelLength * numOfFeatures))
    for row, col in zip(features,label):
        probPredictions = calculateProbabilities(row, weights, labelLength)

        for k in range(labelLength):
            if (k == col):
                cnst = 1
            else:
                cnst = 0
            val = cnst - float(probPredictions[k])
            for p in range(numOfFeatures):
                gradients[k * numOfFeatures +p] -= float(val * row[p]) 
    return array(gradients)

#predict on testing Set
def testingClassification(weights, features, label,  labelLength):
    lengthTestSet = len(label)
    likelihood = 0
    error = 0
    confusionMatrix = numpy.zeros(shape=(labelLength, labelLength)) #initialize confusion matrix
    for row, col in zip(features,label):
        probPredictions = calculateProbabilities(row, weights, labelLength)
        
        maxProb = max(probPredictions)
        index = probPredictions.index(maxProb)
        
        #store predictions in confusion matrix
        confusionMatrix[col][index] += 1
                
        if index != col:
            error += 1
            
    #calculate error rate
    errorRate = float(error) / float(lengthTestSet)
    print "errorRate: ", errorRate
    print "Confusion Matrix: ", confusionMatrix
                 
main()