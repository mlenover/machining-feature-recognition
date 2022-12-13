import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torchdata.datapipes as dp

device = "cuda" if torch.cuda.is_available else "cpu"
print(f"Using {device} device")


FOLDER = '/data/'
datapipe = dp.iter.FileLister([FOLDER]).filter(filter_fn=lambda filename: filename.endswith('.csv'))
datapipe = dp.iter.FileOpener(datapipe, mode='rt')
datapipe = datapipe.parse_csv(delimiter=',')
N_ROWS = 1876  # total number of rows of data
train, valid = datapipe.random_split(total_length=N_ROWS, weights={"train": 0.5, "valid": 0.5}, seed=0)                                            

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
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
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(2):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data