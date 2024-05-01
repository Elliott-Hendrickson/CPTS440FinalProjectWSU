import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from DecisionTreeRegressor import DecisionTree
from RandomForest import RandomForest
from sklearn.metrics import mean_squared_error

#read in data
df = pd.read_csv('train.csv')
#select data set
df = df[['LotArea','OverallQual','OverallCond','YearBuilt', 'GrLivArea', 'TotRmsAbvGrd','YrSold','SalePrice']]
#df.head(30)
#Select feature set
X = df.iloc[:, :-1].values
#select target set
Y = df.iloc[:, -1].values.reshape(-1,1)

decisionTreeRMSE =0
forestRMSE = 0

# runs through this number of iterations
runAmount= 5
for i in range(runAmount):
    print(i)
    #get train / test split
    xTrain, xTest, yTrain, yTest = train_test_split(X,Y, test_size=.1)
    #make decision tree
    treeModel = DecisionTree(minSamplesSplit=2,maxDepth=4)
    #fit decision tree
    treeModel.fit(xTrain, yTrain)
    #make prediction
    pred = treeModel.predict(xTest)
    #get RMSE of decision tree
    decisionTreeRMSE += np.sqrt(mean_squared_error(yTest,pred))
    #make random forest
    randomForest = RandomForest(nTrees=10, maxDepth=4,minSamplesSpit=2)
    #fit random forest to data
    randomForest.fit(xTrain,yTrain)
    #make prediction
    randomForestPrediction = randomForest.predict(xTest)
    #random forest RMSE
    forestRMSE += np.sqrt(mean_squared_error(yTest,randomForestPrediction))


print('Average RMSE for decision tree', (decisionTreeRMSE/runAmount))
print('Average RMSE for forest', (forestRMSE/runAmount))