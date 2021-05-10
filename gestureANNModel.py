import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing
import math
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import joblib

#classes 0 Fist=down, 1 HandRight = right, 2 HandUp= up, 3 HandLeft = left ?
# 0 left? and 1 right?
class MyDataset(Dataset):
    def __init__(self, root):
        self.df = pd.read_csv(root)


    def __getitem__(self, idx):
        y = self.df['class']
        x = self.df.drop(columns=['class'])

        # normalize
        xval = x.values
        min_max_scaler = preprocessing.MinMaxScaler()
        #save scaler
        scaler_filename = "models/min_max_scaler.save"
        xval_scaled = min_max_scaler.fit_transform(xval)
        joblib.dump(min_max_scaler, scaler_filename)

        # convert to torch tensors
        xval_scaled = pd.DataFrame(xval_scaled)
        tensorX = torch.tensor(xval_scaled.to_numpy()).float()
        tensorY = torch.tensor(y.to_numpy()).float()#, dtype=torch.int64
        return tensorX[idx], tensorY[idx]

    def __len__(self):
        return self.df.shape[0]

class Net(nn.Module):

    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 38)
        self.fc2 = nn.Linear(38, 16)
        self.fc3 = nn.Linear(16, 1)
        # self.fc1 = nn.Linear(n_features, 12)
        # self.fc2 = nn.Linear(12, 2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        #x = self.softmax(x)
        ##x= F.relu(self.fc3(x))
        #x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc3(x))

        return x


def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)

# trainData = MyDataset('train30.csv')
# testData = MyDataset('test30.csv')
##combined trainset 3 with 30
trainData = MyDataset('CSV/train3.csv')
testData = MyDataset('CSV/test3.csv')

train_loader = DataLoader(dataset=trainData, batch_size=1, shuffle =True)
test_loader = DataLoader(dataset=testData, batch_size=1, shuffle =True)

num_epochs = 2

#42 features input
net = Net(38)

criterion = nn.BCELoss()
#criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.0001)
torch.manual_seed(0)
for epoch in range(num_epochs):
    train_correct = 0
    print(f'EPOCH : {epoch + 1}')
    accuracy = 0
    for i, (inputs, labels) in enumerate(train_loader, 0):

        #inputs, labels = Variable(inputs), Variable(labels).long()
        inputs, labels = Variable(inputs).float(), Variable(labels).float()

        net.train()
        optimizer.zero_grad()
        y_pred = net(inputs)
        print(y_pred.item())
        if y_pred.item() > 0.5:
            n = 1
        else:
            n = 0
        if n == labels.item():
            train_correct +=1
        labels = labels.unsqueeze(1)
        loss = criterion(y_pred, labels)

        loss.backward()

        optimizer.step()

        max_value = torch.max(y_pred).item()
        # n = 0
        # for j in range(len(y_pred[0])):
        #     if (y_pred[0][j].item() == max_value):
        #         n = j
        # if n == labels.item():
        #     train_correct += 1
        accuracy = train_correct / (i + 1)

        print(
            f'data row: {i + 1}, prediction:{n}, truth:{labels.item()}, accuracy:{accuracy * 100}%, loss:{round_tensor(loss)}')
    #print(f'train accuracy: {accuracy*100}%')


    test_correct = 0
    total = 0
    n=0
    with torch.no_grad():

        for data in test_loader:

            inputs, labels = data
            net.eval()
            outputs = net(inputs)
            #
            if outputs.item() > 0.5:
                n = 1
            else:
                n = 0

            if n == labels.item():
                test_correct +=1
            accuracy = test_correct / (total+1)
            total += labels.size(0)
            print(f'TEST RESULTS: outputs:{outputs}, predicted: {n}, truth:{labels.item()} ')


    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         test_correct += (predicted == labels).sum().item()
    #
    # print('Test Accuracy: %d %%' % (
    #         100 * test_correct / total))
    print(f'test Accuracy {int(accuracy * 100)}%')

#save model
PATH = 'models/gestureModel_net.pth'
torch.save(net.state_dict(), PATH)

## To load model!
## net.load_state_dict(torch.load(PATH))