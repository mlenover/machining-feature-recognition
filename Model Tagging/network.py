import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torchdata.datapipes as dp
import csv
import numpy as np

device = "cuda" if torch.cuda.is_available else "cpu"
print(f"Using {device} device")

data = []

FOLDER = '/data/'
with open(os.getcwd() + FOLDER + 'Data File.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

data = np.array(data).astype(float)

labels = data[:,0].flatten()
labels = torch.from_numpy(labels)
labels = labels.to(device=device, dtype=torch.long)

train_data = data[:,1:]
train_data = np.rot90(train_data)
train_data = torch.from_numpy(data[:,1:])
train_data = train_data.to(device=device, dtype=torch.float32)

datapipe = dp.iter.FileLister([FOLDER]).filter(filter_fn=lambda filename: filename.endswith('.csv'))
datapipe = dp.iter.FileOpener(datapipe, mode='rt')
datapipe = datapipe.parse_csv(delimiter=',')
N_ROWS = 1876  # total number of rows of data
train, valid = datapipe.random_split(total_length=N_ROWS, weights={"train": 0.5, "valid": 0.5}, seed=0)                                            

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(61, 35),
            nn.ReLU(),
            nn.Linear(35, 50),
            nn.ReLU(),
            nn.Linear(50, 35),
            nn.ReLU(),
            nn.Linear(35, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 17)
        )
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        #logits = nn.functional.Softmax(logits, dim=1)
        return logits

model = NeuralNetwork().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(100):
    
    running_loss = 0.0
    correct = 0
    
    for i, sample in enumerate(train_data):
        outputs = model(sample)
        loss = criterion(outputs, labels[i])
        running_loss = running_loss + loss.item()
        
        estimatedclass = torch.argmax(outputs, dim=0).item()
        actualclass = labels[i].item()
        
        if estimatedclass == actualclass:
            correct = correct + 1
        
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d]' %(epoch+1, 10, i+1))
    
    print('Epoch complete, total loss : %.4f, accuracy : %.2f %%' %(running_loss, correct*100/(i+1)))