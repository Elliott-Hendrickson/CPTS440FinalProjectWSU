import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from DecisionTreeRegressor import DecisionTree
from RandomForest import RandomForest
from sklearn.metrics import mean_squared_error

df = pd.read_csv('train.csv')
normalized = pd.read_csv('transformed_data.csv')
#df.head()
y = df['SalePrice']

X = df[['LotArea','OverallQual','OverallCond','YearBuilt']]
#X = X.drop(X[X['LotFrontage'] == 'NaN'].index)
#mask = X['LotFrontage'] == 'NA'
#X=X[~mask]
#X['LotFrontage'] = X['LotFrontage'].replace('',0)
#print(X.head(30))

#X = df.iloc[:, :-1].values
#y = df.iloc[:, -1].values.reshape(-1,1)
#print(X)
#print(y)
df = df[['LotArea','OverallQual','OverallCond','YearBuilt', 'GrLivArea', 'TotRmsAbvGrd','YrSold','SalePrice']]
normalized = normalized[['LotArea','OverallQual','OverallCond','YearBuilt', 'GrLivArea', 'TotRmsAbvGrd', 'YrSold','SalePrice']]
df.head(30)
X = df.iloc[:, :-1].values

Y = df.iloc[:, -1].values.reshape(-1,1)
xTrain, xTest, yTrain, yTest = train_test_split(X,Y, test_size=.1)
treeModel = DecisionTree(minSamplesSplit=2,maxDepth=3)
#s = treeModel.buildTree(df)
treeModel.fit(xTrain, yTrain)

treeModel.print_tree()

pred = treeModel.predict(xTest)
#print(pred)
#print(yTest)
#print(mean_squared_error(yTest,pred))
print(np.sqrt(mean_squared_error(yTest,pred)))
#print(s)

randomForest = RandomForest(nTrees=10, maxDepth=3,minSamplesSpit=2)
randomForest.fit(xTrain,yTrain)
randomForestPrediction = randomForest.predict(xTest)
print(np.sqrt(mean_squared_error(yTest,randomForestPrediction)))

accuracyDecisionTree = (np.sum(pred) / np.sum(yTest))
accuracyRandomForest = (np.sum(randomForestPrediction) / np.sum(yTest)) 
print(accuracyDecisionTree)
print(accuracyRandomForest)
decisionTreeRMSE =0
forestRMSE = 0

for i in range(50):
    print(i)
    xTrain, xTest, yTrain, yTest = train_test_split(X,Y, test_size=.1)
    #treeModel = DecisionTree(minSamplesSplit=2,maxDepth=10)
    #treeModel.fit(xTrain, yTrain)
    #pred = treeModel.predict(xTest)
    #decisionTreeRMSE += np.sqrt(mean_squared_error(yTest,pred))
    randomForest = RandomForest(nTrees=20, maxDepth=6,minSamplesSpit=2)
    randomForest.fit(xTrain,yTrain)
    randomForestPrediction = randomForest.predict(xTest)
    forestRMSE += np.sqrt(mean_squared_error(yTest,randomForestPrediction))


runAmount = 50
print('Average RMSE for decision tree', (decisionTreeRMSE/runAmount))
print('Average RMSE for forest', (forestRMSE/runAmount))