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
        for i in range(self.nTrees):
            tree = DecisionTree(minSamplesSplit=2,maxDepth=3)
            X, y = self.bootstrap(X,y)
            tree.fit(X,y)
            self.trees.append(tree)
        
            
    def bootstrap(self,X,y):
        numSamples = np.shape(X)[0]
        ids = np.random.choice(numSamples,numSamples,replace=True)
        return X[ids], y[ids]
    
    def predict(self, X):
        predictions = []
        finalPredictions =[]
        for tree in self.trees:
            predictions.append(tree.predict(X))
        sum = 0
        for i in range((len(X))):
            for j in range(len(predictions)):
                sum += predictions[j][i]
            finalPredictions.append(sum/self.nTrees)
            sum = 0
        return finalPredictions
