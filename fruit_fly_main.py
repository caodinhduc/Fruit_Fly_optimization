import sys
import random
import math
#import numpy as np
import pandas as pd
#from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.utils import shuffle


def readFloodDataset():
    dataset = pd.read_csv('train.csv')
    data = dataset.iloc[:,1: ].values
    target = dataset.iloc[:,0].values
    return (data,target)

class fly:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

def fitness(data_scaled, target, x,y):
    clf = svm.SVC(kernel='rbf', C=x, gamma=y)
    scores = cross_val_score(clf,data_scaled, target, cv=10)
    print(scores)   
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)
    return scores.mean()

def run(data_scaled, target, population, x_axis1, y_axis1, bestSmell, x_axis2, y_axis2):
    
    flies = []
    # generate the population
    for i in range(0,population):
        
        xi1 = x_axis1 + random.uniform(-1,1)
        yi1 = y_axis1 + random.uniform(-1,1)
        xi2 = x_axis2 + random.uniform(-1,1)
        yi2 = y_axis2 + random.uniform(-1,1)
        
        
        flies.append(fly(xi1,yi1,xi2,yi2))
        
    minSmell =  -sys.maxsize
    bestIndex = -1
    
    for currFly in flies:
        
        d1 = math.sqrt(math.pow(currFly.x1,2)+math.pow(currFly.y1,2))
        d2 = math.sqrt(math.pow(currFly.x2,2)+math.pow(currFly.y2,2))
        
        d1_inv = round(1/d1,2)
        d2_inv = round(1/d2,2)
        
        # limit range
        d1_inv = min(d1_inv,math.pow(2,15))
        d1_inv = max(d1_inv,math.pow(2,-5))
        
        d2_inv = min(d2_inv,2)
        d2_inv = max(d2_inv,math.pow(2,-15))
        

        # return score
        smell = fitness(data_scaled, target, d1_inv, d2_inv)
        if(minSmell < smell):
            minSmell = smell
            bestIndex = flies.index(currFly)
    
    bestSmell = minSmell
    x_axis1 = flies[bestIndex].x1
    y_axis1 = flies[bestIndex].y1
    x_axis2 = flies[bestIndex].x2
    y_axis2 = flies[bestIndex].y2
    
    return (x_axis1, y_axis1, bestSmell, x_axis2, y_axis2)


def main():
    population = 10
    x_axis1 = random.uniform(0,1)
    y_axis1 = random.uniform(0,1)
    x_axis2 = random.uniform(0,1)
    y_axis2 = random.uniform(0,1)
    bestSmell = sys.maxsize
    print('bestSmell: ', bestSmell)
    max_iter = 100
    
    data, target = readFloodDataset()
    data_scaled = preprocessing.scale(data)
    
    for i in range(max_iter):
        print(f'Running iteration {i+1} ...')
        x_axis1, y_axis1, bestSmell, x_axis2, y_axis2 = run(data_scaled, target, population, x_axis1, y_axis1, bestSmell, x_axis2, y_axis2)
        print(round((1/math.sqrt(math.pow(x_axis1,2)+math.pow(y_axis1,2))),2), round((1/math.sqrt(math.pow(x_axis2,2)+math.pow(y_axis2,2))),2), bestSmell)

main()
# train svc with the best params
# 1.9 0.08 0.99