import numpy as np
import pandas as pd

class Node:
    def __init__(self, featureIndex=None, threshold=None, left=None, right=None, varianceReduction = None,value=None):
        self.featureIndex = featureIndex
        self.threshold = threshold
        self.left = left
        self.right = right
        self.varianceReduction = varianceReduction
        self.value=value
        
    
class DecisionTree:
    def __init__(self, minSamplesSplit=2, maxDepth=2):
        self.minSamplesSplit=minSamplesSplit
        self.maxDepth=maxDepth
        self.root=None
        
    def buildTree(self, df, currentDepth=0):
        bestSplitNode={}
        Xdata, ydata = df[:,:-1], df[:,-1]
        numSamples, numFeatures = np.shape(Xdata)
        #split condisions
        if numSamples >= self.minSamplesSplit and currentDepth <= self.maxDepth:
            #find best split
            bestSplitNode = self.getBestSplit(Xdata, ydata, df)
            # create left and right nodes
            if bestSplitNode['varianceReduction'] > 0:
                lSubTree = self.buildTree(bestSplitNode['dataLeft'], currentDepth + 1)
                rSubTree = self.buildTree(bestSplitNode['dataRight'], currentDepth + 1)
                return Node(bestSplitNode['featureIndex'], bestSplitNode['threshold'], lSubTree, rSubTree, bestSplitNode['varianceReduction'])
        
        #value for leaf node (prediction) 
        value = np.mean(ydata)
        return Node(value=value)
            
    def getBestSplit(self, X,y, df):
        bestSplitNode = {}
        bestSplitNode['varianceReduction'] = 0

        numSamples, numFeatures = np.shape(X)
        maxVariance = float('-inf')
        minVariance = float('inf')
        leftList =[]
        rightList =[]
        #iterates through features
        for i in range(numFeatures):
            values = X[:,i]
            uniqueValues = np.unique(values)
            #iterates through all posible thresholds for that feature in the data set
            for threshold in uniqueValues:
                queryFor = str(threshold)
                #splits the data at threshold
                dataLeft = df[df[:, i] <= threshold]
                dataRight = df[df[:, i] > threshold]
                
                if len(dataLeft) >0 and len(dataRight) > 0:
                    leftY = dataLeft[:,-1]
                    rightY = dataRight[:,-1]
                    #finds the variance of the threshold
                    currVariance = self.varianceReduction(y,leftY, rightY)
                    #the node with the highest variance will be returned
                    if currVariance > maxVariance:
                        maxVariance = currVariance
                        bestSplitNode['featureIndex'] = i
                        bestSplitNode['threshold'] = threshold
                        bestSplitNode['dataLeft'] = dataLeft
                        bestSplitNode['dataRight'] = dataRight
                        bestSplitNode['varianceReduction'] = currVariance
                    
        return bestSplitNode
    
    def varianceReduction(self, parent, lChild, rChild):
        weightL = len(lChild) / len(parent)
        weightR = len(rChild) / len(parent)
        
        return np.var(parent) - (weightL * np.var(lChild) + weightR * np.var(rChild))
    
    def fit(self, X, y):
        #puts data back into a 2d array
        data = np.concatenate((X,y), axis=1)
        #sets root
        self.root = self.buildTree(data)
        
    def makePrediction(self, x, tree):
        #traverses through tree based on the values of the features in x
        
        if tree.value!=None: return tree.value
        featureValue = x[tree.featureIndex]
        if featureValue <= tree.threshold:
            return self.makePrediction(x, tree.left)
        else:
            return self.makePrediction(x, tree.right)
        
    def predict(self, X):
        prediction = [self.makePrediction(x, self.root) for x in X]
        
        return prediction
    
    def print_tree(self, tree=None, indent=" "):
        
        if not tree:
            tree = self.root
            if tree is None:
                print("The tree has not been built or is empty.")
                return
            print('Root node here')
        if tree.value is not None:
            print(tree.value)
        else:
            print(f"X_"+str(tree.featureIndex), "<=", tree.threshold, "?", tree.varianceReduction)
            print(f"%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print(f"%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

