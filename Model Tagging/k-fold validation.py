import torch
from torch import nn
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from sklearn.model_selection import KFold

device = "cuda" if torch.cuda.is_available else "cpu"

class DatasetFromCSV():
    def __init__(self, csv_path):
        dataset = pd.read_csv(csv_path)
        dataset = np.array(dataset).astype(int)
        
        labels = dataset[:,0].flatten()
        labels = torch.from_numpy(labels)
        labels = labels.to(device=device, dtype=torch.long)
        self.labels = labels
        
        data = torch.from_numpy(dataset[:,1:])
        data = data.to(device=device, dtype=torch.float32)
        self.data = data
        
        
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        
        return sample, label

    def __len__(self):
        return len(self.labels)
    
    def move_to_gpu(self):
        self.labels = self.labels.to(device=device, dtype=torch.long)
        self.data = self.data.to(device=device, dtype=torch.float32)

def held_out_split_dataset(dataset, held_out_split, random_seed):
    shuffle_dataset = True
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(held_out_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    held_out_indices, train_indices = indices[:split], indices[split:]

    return train_indices, held_out_indices

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset parameters'):
            layer.reset_parameters()

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
        return logits

if __name__ == '__main__':
    
    batch_size = 1
    random_seed = 0
    
    test_split = 0.2
    
    k_folds = 5
    num_epochs = 500
    loss_function = nn.CrossEntropyLoss()
    
    results = {}
    
    torch.manual_seed(random_seed)
    
    csv_path = "data/Data File - April 17 Added Non Feature Data.csv"
    dataset = DatasetFromCSV(csv_path)

    train_indices, held_out_indices = held_out_split_dataset(dataset, test_split, random_seed)
    
    test_sampler = SubsetRandomSampler(held_out_indices)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)

    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset[train_indices][0])):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
        
        train_loader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=1, sampler=train_subsampler)
        valid_loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=1, sampler=valid_subsampler)
    
        model = NeuralNetwork()
        model.apply(reset_weights)
        model = NeuralNetwork().to(device)
        
        optimizer = optim.Adam(model.parameters(),lr=1e-4)
        
        for epoch in range(0, num_epochs):

            # Print epoch
            print(f'Starting epoch {epoch+1}')
      
            # Set current loss value
            current_loss = 0.0
      
            # Iterate over the DataLoader for training data
            for i, data in enumerate(train_loader, 0):
                
                # Get inputs
                inputs, targets = data
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Perform forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = loss_function(outputs, targets)
                
                # Perform backward pass
                loss.backward()
                
                # Perform optimization
                optimizer.step()
                
                # Print statistics
                current_loss += loss.item()
            
            print('Loss at end of epoch: %.3f' %(current_loss / i + 1))
        
        # Process is complete.
        print('Training process has finished. Saving trained model.')
    
        # Print about testing
        print('Starting testing')
        
        # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        torch.save(model.state_dict(), save_path)
    
        # Evaluationfor this fold
        correct, total = 0, 0
        with torch.no_grad():
    
            # Iterate over the test data and generate predictions
            for i, data in enumerate(valid_loader, 0):
              
                  # Get inputs
                  inputs, targets = data
            
                  # Generate outputs
                  outputs = model(inputs)
            
                  # Set total and correct
                  _, predicted = torch.max(outputs.data, 1)
                  total += targets.size(0)
                  correct += (predicted == targets).sum().item()
              
            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')