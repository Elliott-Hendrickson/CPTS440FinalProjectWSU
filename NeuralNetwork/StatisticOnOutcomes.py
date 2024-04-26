import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
import pandas as pd
def readOutcomes():
    with open("generationThreePredictions.txt", "r") as file:
        content_list = file.readlines()
    return(content_list)
def trimAndConvert(listOfOutputs):
    sumOfPredictions, sumOfRealValues = 0,0
    averageDifference = 0
    for output in listOfOutputs:
        localReal, LocalPrediction = output.replace("R:", "").replace("\n", "").split(",P:")
        averageDifference = (averageDifference + (abs(float(localReal)-float(LocalPrediction)))/float(localReal))/2#"""/float(localReal)"""
    print(averageDifference)

def fixTransformedData():
    df = pd.read_csv('transformed_data.csv')
    sf = pd.read_csv('train.csv')
    df["SalePrice"] = sf["SalePrice"]
    df.to_csv('transformed_data.csv', index=False)
trimAndConvert(readOutcomes())
