import numpy as np
import pandas as pd
from DecisionTreeRegressor import DecisionTree

class RandomForest:
    def __init__(self, nTrees=5, maxDepth=3, minSamplesSpit=2, nFeatures=None):
        self.nTrees=nTrees
        self.maxDepth =maxDepth
        self.minSamplesSplit = minSamplesSpit
        self.nFeatures=nFeatures
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        #iterates through trees specified when the random forest was substantiated
        for i in range(self.nTrees):
            #makes decision tree
            tree = DecisionTree(minSamplesSplit=2,maxDepth=3)
            #bootstraps the data
            X, y = self.bootstrap(X,y)
            #fits the tree to the data
            tree.fit(X,y)
            #adds it to the list of trees in forest
            self.trees.append(tree)
        
            
    def bootstrap(self,X,y):
        #gets number of samples
        numSamples = np.shape(X)[0]
        #randomizes the data set
        ids = np.random.choice(numSamples,numSamples,replace=True)
        return X[ids], y[ids]
    
    def predict(self, X):
        predictions = []
        finalPredictions =[]
        #iterates through the trees in forest
        for tree in self.trees:
            predictions.append(tree.predict(X))
        sum = 0
        #averages the predictions made by all the trees in the forest
        for i in range((len(X))):
            for j in range(len(predictions)):
                sum += predictions[j][i]
            finalPredictions.append(sum/self.nTrees)
            sum = 0
        return finalPredictions