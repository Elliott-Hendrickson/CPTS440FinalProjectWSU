
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('transformed_data.csv').dropna()

#["LotArea", "OverallCond", "YearBuilt", "BedroomAbvGr", "YrSold", "1stFlrSF", "2ndFlrSF", "SalePrice"]


subsetTestDF = df[["SalePrice"]]
subsetTrainDF = df[["OverallQual", "YearBuilt", "YearRemodAdd", "MasVnrArea", "TotalBsmtSF", "1stFlrSF", "GrLivArea", "FullBath", "TotRmsAbvGrd", "GarageCars", "GarageArea", "ExterQual_TA", "BsmtQual_Ex", "KitchenQual_Ex"]]
print(subsetTrainDF.shape)

xtrain_df, xtest_df = train_test_split(subsetTrainDF, test_size=0.20, random_state=42)
ytrain_df, ytest_df = train_test_split(subsetTestDF, test_size=0.20, random_state=42)
X_train_np=xtrain_df.to_numpy()
Y_train_np=ytrain_df.to_numpy()
X_test_np =xtest_df.to_numpy()

Y_test_np=ytest_df.to_numpy()

train_dataset = TensorDataset(torch.tensor(X_train_np,dtype=torch.float),
                              torch.tensor(Y_train_np.reshape((-1,1)),dtype=torch.float))
train_dataloader = DataLoader(train_dataset, batch_size = 1300)

test_dataset = TensorDataset(torch.tensor(X_test_np,dtype=torch.float),
                              torch.tensor(Y_test_np.reshape((-1,1)),dtype=torch.float))
test_dataloader = DataLoader(test_dataset, batch_size = 1300)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.hidden_layer_1 = nn.Linear(14,196)
        self.hidden_activation = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(196,784)
        self.hidden_activation = nn.ReLU()
        self.hidden_layer_3 = nn.Linear(784,64)
        self.hidden_activation = nn.ReLU()  
        self.out = nn.Linear(64,1)
    def forward(self,x):
        x = self.hidden_layer_1(x)
        x= self.hidden_activation(x)
        x = self.hidden_layer_2(x)
        x= self.hidden_activation(x)
        x = self.hidden_layer_3(x)
        x= self.hidden_activation(x)
        x = self.out(x)
        return x
model = NeuralNet().to('cpu')

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = 0
    for i, (X,y) in enumerate(dataloader):
        X,y = X.to('cpu'), y.to('cpu')

        y_hat = model(X)
        mse = loss_fn(y_hat,y)
        train_loss += mse.item()

        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
    num_batches = len(dataloader)
    train_mse = train_loss / num_batches
    print(f'Train RMSE: {train_mse**(1/2)}')

def test(dataLoader, model, loss_fun):
    model.eval()
    test_loss= 0
    with torch.no_grad():
        for X, y in dataLoader:
            X, y = X.to('cpu'), y.to('cpu')
            y_hat = model(X)
            test_loss += loss_fn(y_hat, y).item()
    num_batches = len(dataLoader)
    test_mse = test_loss / num_batches

    print(f'Test RMSE: {test_mse**(1/2)}')

epochs = 50000
for epoch in range(epochs):
    print(f"Epoch { epoch+1}:")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)
with open('TrainedNN.pt', 'wb') as f: 
    torch.save(model.state_dict(), f) 

with open('TrainedNN.pt', 'rb') as f: 
    model = NeuralNet()
    model.load_state_dict(torch.load(f))  
    model.eval()

def findStatistics(dataLoader, model):
    model.eval()
    outputList = []
    with torch.no_grad():
        for X, y in dataLoader:
            X= X.to('cpu')
            predictedPrice = model(X)
            outputList.append(f"R:{y.item()},P:{predictedPrice.item()}")
    
        with open("output.txt", "w") as file:
            for predictionStatistic in outputList:
                file.write(predictionStatistic + "\n")      
findStatistics(train_dataset,model)