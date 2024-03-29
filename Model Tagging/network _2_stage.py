import torch
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from crossover import generate_data
from ID3_classifier import minimize_class_imbalance_id3, id3_estimate
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = "cuda" if torch.cuda.is_available else "cpu"
print(f"Using {device} device")

do_id3_tree = True
do_crossover = True
do_remove_duplicates = False
do_multi_stage_network = False

#import the csv
FOLDER = "data/"
data = pd.read_csv(FOLDER + "Data File - April 03 Replaced Fillets.csv")
header = list(data.head(0))
num_headers = len(header)
header_dict = dict(zip(header, range(0,num_headers)))

#count the number of samples in each class
classes = np.array(data["Class"]).astype(int)
classes = np.sort(classes)
class_dict = {}

for i in classes:
    class_dict[i] = class_dict.get(i, 0) + 1

greatest_class = max(class_dict, key=class_dict.get)
greatest_num_samples = max(class_dict.values())

#graph a histogram of number of samples of each class
a0 = plt.figure(0)
plt.bar(class_dict.keys(), class_dict.values())
plt.pause(0.05)

if(do_id3_tree):
    sorted_class_dict = sorted(class_dict.values())
    num_classes = len(sorted_class_dict)
    second_greatest_class_samples = sorted_class_dict[num_classes-2]
    id3_tree = minimize_class_imbalance_id3(data, over_rep_class=None, max_depth=7, max_class_samples=None)


np_data = np.array(data[1:]).astype(int)
#generate some new data using crossover
if(do_crossover):
    unique_data, generated_data = generate_data(np_data)
    augmented_data = np.concatenate((unique_data, generated_data))
    
    #count number of samples in each class in augmented data
    augmented_data = np.array(augmented_data).astype(int)
    aug_classes = augmented_data[:,0].astype(int)
    aug_classes = np.sort(aug_classes)
    aug_class_dict = {}
    
    for i in aug_classes:
        aug_class_dict[i] = aug_class_dict.get(i, 0) + 1
    data = augmented_data

#graph a histogram of number of samples of each class in augmented data
#a1 = plt.figure(1)
#plt.bar(aug_class_dict.keys(), aug_class_dict.values())
#plt.pause(0.05)

#remove duplicates from augmented data

if(do_remove_duplicates):
    augmented_data_no_duplicates = np.unique(data, axis=0).astype(int)
    
    no_dup_aug_classes = augmented_data_no_duplicates[:,0].astype(int)
    no_dup_aug_classes = np.sort(no_dup_aug_classes)
    no_dup_aug_class_dict = {}

    for i in no_dup_aug_classes:
        no_dup_aug_class_dict[i] = no_dup_aug_class_dict.get(i, 0) + 1
    
    data = augmented_data_no_duplicates

#a2 = plt.figure(2)
#plt.bar(no_dup_aug_class_dict.keys(), no_dup_aug_class_dict.values())
#plt.pause(0.05)

num_rows_removed = 0
if(do_id3_tree):
    tree_decisions_removed = None
    
    for i, sample in enumerate(data):
        
        sample_dict = dict(zip(header_dict.keys(), sample))
        
        class_estimate = id3_estimate(id3_tree, sample_dict)
            
        if class_estimate == -1:
            #we know the sample is NOT the overrepresented one, so let's add it to a new list
            if tree_decisions_removed is None:
                tree_decisions_removed = sample
            else:
                tree_decisions_removed = np.vstack((tree_decisions_removed, sample))
        else:
            num_rows_removed = num_rows_removed+1
    
    data = tree_decisions_removed

train_data = data
labels = train_data[:,0].flatten()

if do_multi_stage_network:
    none_class_labels = labels.copy()
    
    for i, label in enumerate(labels):
        if label != 0:
            none_class_labels[i] = 1

    none_class_labels = torch.from_numpy(none_class_labels)
    none_class_labels = none_class_labels.to(device=device, dtype=torch.long)
        
labels = torch.from_numpy(labels)
labels = labels.to(device=device, dtype=torch.long)

#train_data = data[:,1:]
#train_data = np.rot90(train_data)
train_data = torch.from_numpy(train_data[:,1:])
train_data = train_data.to(device=device, dtype=torch.float32)

#datapipe = dp.iter.FileLister([FOLDER]).filter(filter_fn=lambda filename: filename.endswith('.csv'))
#datapipe = dp.iter.FileOpener(datapipe, mode='rt')
#datapipe = datapipe.parse_csv(delimiter=',')
#N_ROWS = 162  # total number of rows of data
#train, valid = datapipe.random_split(total_length=N_ROWS, weights={"train": 0.5, "valid": 0.5}, seed=0)                                            

if do_multi_stage_network:
    class NoneFeatureNetwork(nn.Module):
        def __init__(self):
            super(NoneFeatureNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(61, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 4),
                nn.ReLU(),
                nn.Linear(4, 2)
            )
            
        def forward(self, x):
    
            logits = self.linear_relu_stack(x)
            return logits
    
    model1 = NoneFeatureNetwork().to(device)
    print(model1)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model1.parameters())

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

model2 = NeuralNetwork().to(device)
print(model2)

criterion = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(model2.parameters())

a2 = plt.figure(2)
plt.axis()
completed_epochs = []
loss_history = []
accuracy_history = []

m = nn.Dropout(p=0.2)

if do_multi_stage_network:
    for epoch in range(50):
        completed_epochs.append(epoch)
        running_loss = 0.0
        correct = 0
        accuracy = 0.0
        
        for i, sample in enumerate(train_data):       
            outputs = model1(m(sample))
            
            loss = criterion(outputs, none_class_labels[i])
            running_loss = running_loss + loss.item()
            
            estimatedclass = torch.argmax(outputs, dim=0).item()
            actualclass = none_class_labels[i].item()
            
            if estimatedclass == actualclass:
                correct = correct + 1
            
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d]' %(epoch+1, 10, i+1))
                
        accuracy = running_loss, correct*100/(i+1)
        accuracy = (correct)*100/(i+1)
        print('Epoch complete, total loss : %.4f, network accuracy : %.2f %%' %(running_loss, accuracy))
        
        loss_history.append(running_loss)
        accuracy_history.append(accuracy)
        
        plt.clf()
        plt.plot(completed_epochs, accuracy_history)
        #plt.ylim(0, 5000)
        plt.pause(0.05)

for epoch in range(50):
    completed_epochs.append(epoch)
    running_loss = 0.0
    correct = 0
    accuracy = 0.0
    
    for i, sample in enumerate(train_data):
        #if (labels[i] != 0 and do_multi_stage_network) or not do_multi_stage_network:
        if not do_multi_stage_network:
            outputs = model2(m(sample))
            loss2 = criterion(outputs, labels[i])
            running_loss = running_loss + loss2.item()
            
            estimatedclass = torch.argmax(outputs, dim=0).item()
            actualclass = labels[i].item()
            
            print(estimatedclass)
            
            if estimatedclass == actualclass:
                correct = correct + 1
            
            loss2.backward()
            optimizer2.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d]' %(epoch+1, 10, i+1))
            
    #accuracy = running_loss, correct*100/(i+1)
    accuracy = (correct)*100/(i+1)
    print('Epoch complete, total loss : %.4f, network accuracy : %.2f %%' %(running_loss, accuracy))
    
    loss_history.append(running_loss)
    accuracy_history.append(accuracy)
    
    plt.clf()
    plt.plot(completed_epochs, accuracy_history)
    #plt.ylim(0, 5000)
    plt.pause(0.05)

plt.show()

print('Overall accuracy : %.2f %%' %((correct+num_rows_removed)*100/(i+1+num_rows_removed)))

test_data = pd.read_csv(FOLDER + "Test Data File.csv")

nn.Dropout(p=0.0)

correct = 0
correct_id3 = 0
incorrect = 0
incorrect_id3 = 0

y_pred = []
y_true = []
featureList = ["None Feature", "Simple Hole", "Closed Pocket", "Countersunk Hole", "Opened Pocket", "Counterbore Hole",
                "Closed Island", "Counterdrilled Hole", "Opened Island", "Tapered Hole", "Inner Fillet", "Closed Slot",
                "Outer Fillet", "Opened Slot", "Inner Chamfer", "Floorless Slot", "Outer Chamfer"]

for row in test_data.iterrows():

    sample_dict = row[1].to_dict()
    sample_dict = dict([a, int(x)] for a, x in sample_dict.items())
    
    if do_id3_tree:
        id3_class_estimate = id3_estimate(id3_tree, sample_dict)
    
        if id3_class_estimate != -1:
            estimatedclass = id3_class_estimate
            pass
        
    sample = np.array(row[1][1:]).astype(int)
    sample =  torch.from_numpy(sample)
    sample = sample.to(device=device, dtype=torch.float32)
    
    if do_multi_stage_network:
        outputs = model1(m(sample))
        is_none_network_output = torch.argmax(outputs, dim=0).item()
    
        if is_none_network_output == 0:
            estimateclass = 0
            pass
        
    outputs = model2(m(sample))
    estimatedclass = torch.argmax(outputs, dim=0).item()
    
    ground_truth = np.array(row[1][0]).astype(int)
    
    y_pred.append(featureList[estimatedclass])
    y_true.append(featureList[ground_truth])
    
    if estimatedclass == ground_truth:
        correct = correct + 1
    
        if do_id3_tree:    
            if id3_class_estimate != -1:
                correct_id3 = correct_id3 + 1
    else:
        incorrect = incorrect + 1
        
        if do_id3_tree:
            if id3_class_estimate != -1:
                incorrect_id3 = correct_id3 + 1

accuracy = correct / (correct + incorrect)
accuracy = accuracy * 100
print('Test Data Accuracy Overall: %.4f %%' %(accuracy))

if do_id3_tree:
    id3_accuracy = correct_id3 / (correct_id3 + incorrect_id3)
    id3_accuracy = id3_accuracy * 100
    print('Test Data ID3 Accuracy: %.4f %%' %(id3_accuracy))

cm = confusion_matrix(y_true, y_pred, labels=featureList)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=featureList)
disp.plot(xticks_rotation=90)
plt.show()