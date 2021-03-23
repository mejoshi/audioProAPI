#!/usr/bin/env python
# coding: utf-8

# In[28]:


from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

from tempfile import TemporaryFile

import os
import pickle
import random
import operator

import math


# In[29]:


# function to get the distance between feature vecotrs and find neighbors
def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    
    return neighbors


# In[30]:


# identify the class of the instance
def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1

    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)

    return sorter[0][0]


# In[31]:


# function to evaluate the model
def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == prediction[x]:
            correct += 1
    
    return (1.0 * correct) / len(testSet)


# In[32]:


# Split the dataset into training and testing sets respectively
dataset = []

def loadDataset(filename, split, trSet, teSet):
    with open('my.dat', 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])
trainingSet = []
testSet = []
loadDataset('my.dat', 0.66, trainingSet, testSet)


# In[33]:


def distance(instance1 , instance2 , k ):
    distance =0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance



# # making predictions using KNN
# leng = len(testSet)
# print(len)
# predictions = []
# for x in range(leng):
#     predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))

# accuracy1 = getAccuracy(testSet, predictions)
# print(accuracy1)

# test_dir = "Test/"
# test_file = test_dir + "test.wav"
# # test_file = test_dir + "test2.wav"
# # test_file = test_dir + "test4.wav"


# In[40]:

# i = 0
# (rate, sig) = wav.read(test_file)
# mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
# covariance = np.cov(np.matrix.transpose(mfcc_feat))
# mean_matrix = mfcc_feat.mean(0)
# feature = (mean_matrix, covariance, i)


# from collections import defaultdict
# results = defaultdict(int)
#
# directory = "Data/genres_original/"
#
# i = 1
# for folder in os.listdir(directory):
#     results[i] = folder
#     i += 1
#
#
#
#
# pred = nearestClass(getNeighbors(dataset, feature, 5))
# print(results[pred])


def predict(test_file):
    i = 0
    (rate, sig) = wav.read(test_file)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature = (mean_matrix, covariance, i)
    from collections import defaultdict
    results = defaultdict(int)

    directory = "Data/genres_original/"

    i = 1
    for folder in os.listdir(directory):
        results[i] = folder
        i += 1
        pred = nearestClass(getNeighbors(dataset, feature, 5))
    print(results[pred])
    return results[pred]








