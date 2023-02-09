import os
import torch
from torch import nn
import torch.optim as optim
import torchdata.datapipes as dp
import csv
import numpy as np
import matplotlib.pyplot as plt
from crossover import generate_data

device = "cuda" if torch.cuda.is_available else "cpu"
print(f"Using {device} device")

data = []

FOLDER = '/data/'
with open(os.getcwd() + FOLDER + 'Data File.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

unique_data, generated_data = generate_data(data)

data = np.concatenate((unique_data, generated_data))
#data = np.unique(data, axis=0).astype(int)

classes = data[:,0].astype(int)
classes = np.sort(classes)
class_dict = {}

for i in classes:
    class_dict[i] = class_dict.get(i, 0) + 1

# print(class_dict.keys())
# print(class_dict.values())

a0 = plt.figure(0)
plt.bar(class_dict.keys(), class_dict.values())
plt.pause(0.05)

labels = data[:,0].flatten()
labels = torch.from_numpy(labels)
labels = labels.to(device=device, dtype=torch.long)

train_data = data[:,1:]
train_data = np.rot90(train_data)
train_data = torch.from_numpy(data[:,1:])
train_data = train_data.to(device=device, dtype=torch.float32)

#datapipe = dp.iter.FileLister([FOLDER]).filter(filter_fn=lambda filename: filename.endswith('.csv'))
#datapipe = dp.iter.FileOpener(datapipe, mode='rt')
#datapipe = datapipe.parse_csv(delimiter=',')
#N_ROWS = 162  # total number of rows of data
#train, valid = datapipe.random_split(total_length=N_ROWS, weights={"train": 0.5, "valid": 0.5}, seed=0)                                            

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

a1 = plt.figure(1)
plt.axis()
completed_epochs = []
loss_history = []
accuracy_history = []

for epoch in range(200):
    
    completed_epochs.append(epoch)
    running_loss = 0.0
    correct = 0
    accuracy = 0.0
    
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
            
    #accuracy = running_loss, correct*100/(i+1)
    accuracy = correct*100/(i+1)
    print('Epoch complete, total loss : %.4f, accuracy : %.2f %%' %(running_loss, accuracy))
    
    loss_history.append(running_loss)
    accuracy_history.append(accuracy)
    
    plt.clf()
    plt.plot(completed_epochs, accuracy_history)
    #plt.ylim(0, 5000)
    plt.pause(0.05)

plt.show()
pass