#Modified from https://scribe.nixnet.services/geekculture/step-by-step-decision-tree-id3-algorithm-from-scratch-in-python-no-fancy-library-4822bbfdd88f

import pandas as pd #for manipulating the csv data
import numpy as np #for mathematical calculation
import os
#import warnings
#warnings.filterwarnings("error")

def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0] #the total size of the dataset
    
    total_entr = 0
    
    for c in class_list: #for each class in the label
        total_class_count = train_data[train_data[label] == c].shape[0] #number of the class
        if total_class_count != 0:
            total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row) #entropy of the class
            total_entr += total_class_entr #adding the class entropy to the total entropy of the dataset
    
    return total_entr

def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0
    
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0] #row count of class c 
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count #probability of the class
            entropy_class = - probability_class * np.log2(probability_class)  #entropy
        entropy += entropy_class
    return entropy

def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique() #unqiue values of the feature
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value] #filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list) #calculcating entropy for the feature value
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy #calculating information of the feature value
        
    return calc_total_entropy(train_data, label, class_list) - feature_info #calculating

def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label) #finding the feature names in the dataset
                                            #N.B. label is not a feature, so dropping it
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list:  #for each feature in the dataset
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain: #selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature
    
    if max_info_feature is None:
        pass
    
    return max_info_feature, max_info_gain

def generate_sub_tree(feature_name, train_data, label, class_list, max_info_gain):
    
    contains_only_conflicts = False 
    if max_info_gain == 0:
        if len(class_list) > 1:
            contains_only_conflicts = True #if remaining data contain one or more labels but identical feature vectors, we have a problem!
        pass
    
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False) #dictionary of the count of unqiue feature value
    tree = {} #sub tree or node
    
    for feature_value, count in feature_value_count_dict.items():
        feature_value_data = train_data[train_data[feature_name] == feature_value] #dataset with only feature_name = feature_value
        
        assigned_to_node = False #flag for tracking feature_value is pure class or not
        for c in class_list: #for each class
            class_count = feature_value_data[feature_value_data[label] == c].shape[0] #count of class c

            if class_count == count: #count of (feature_value = count) of class (pure class)
                tree[feature_value] = c #adding node to the tree
                train_data = train_data[train_data[feature_name] != feature_value] #removing rows with feature_value
                assigned_to_node = True
        if not assigned_to_node: #not pure class
            
            if contains_only_conflicts:
                tree[feature_value] = "!"
            else:
                tree[feature_value] = "?" #as feature_value is not a pure class, it should be expanded further, 
                                          #so the branch is marking with ?
            
    return tree, train_data

def make_tree(root, prev_feature_value, train_data, label, class_list, current_depth, max_depth):
    
    is_max = False
    
    if max_depth is None:
        pass
    elif current_depth >= max_depth:
        is_max = True
        
    if train_data.shape[0] != 0: #if dataset becomes empty after updating
        max_info_feature, max_info_gain = find_most_informative_feature(train_data, label, class_list) #most informative feature
        
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list, max_info_gain) #getting tree node and updated dataset
        next_root = None
        
        if prev_feature_value != None: #add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else: #add to root of the tree
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        
        for node, branch in list(next_root.items()): #iterating the tree node
            if branch == "?" and not is_max: #if it is expandable
                feature_value_data = train_data[train_data[max_info_feature] == node] #using the updated dataset
                make_tree(next_root, node, feature_value_data, label, class_list, current_depth+1, max_depth) #recursive call with updated dataset
                
def id3(train_data_m, label, max_depth):
    train_data = train_data_m.copy() #getting a copy of the dataset
    tree = {} #tree which will be updated
    class_list = train_data[label].unique() #getting unqiue classes of the label
    make_tree(tree, None, train_data, label, class_list, 0, max_depth) #start calling recursion
    return tree

def remove_rows(variable, value, dataset):
    dataset_copy = dataset
    dataset_copy = dataset_copy[getattr(dataset_copy,variable) != value]
    return dataset_copy

def remove_data_rows_to_depth(tree, dataset, current_depth, max_depth):
    dataset_copy = dataset
    
    for key in tree:
        highest_entropy_var = key
    
    highest_entropy_val_list = tree[highest_entropy_var]
    for key in highest_entropy_val_list:
        value = highest_entropy_val_list[key]
        if type(value) is not dict:
            if dataset_copy is None:
                pass
            dataset_copy = remove_rows(highest_entropy_var, key, dataset_copy)
        else:
            if current_depth < max_depth or dataset_copy is None:
                data_subset = dataset_copy[getattr(dataset_copy,highest_entropy_var) == key]
                dataset_copy = dataset_copy[getattr(dataset_copy,highest_entropy_var) != key]
                data_subset = remove_data_rows_to_depth(value, data_subset, current_depth+1, max_depth)
                dataset_copy = pd.concat([dataset_copy, data_subset])
                
    return dataset_copy

def sanitize_tree(tree):
    for key in tree:
        highest_entropy_var = key
    
    highest_entropy_val_list = tree[highest_entropy_var]
    
    for key, value in highest_entropy_val_list.items():
        if type(value) is not dict:
            if type(value) is str:
                tree[highest_entropy_var][key] = -1
        else:
            sanitize_tree(tree[highest_entropy_var][key])       
    return tree
        
        
def trim_tree_at_depth(tree, current_depth, max_depth):
    for key in tree:
        highest_entropy_var = key
        
    highest_entropy_val_list = tree[highest_entropy_var]
    
    for key, value in highest_entropy_val_list.items():
        if type(value) is dict:
            if current_depth >= max_depth:
                tree[highest_entropy_var][key] = -1
            else:
                trim_tree_at_depth(tree[highest_entropy_var][key], current_depth+1, max_depth)
    return tree
    

def minimize_class_imbalance_id3(train_data, over_rep_class=None, max_depth=None, max_class_samples=None):
    data_copy = train_data.copy()
    
    if over_rep_class is not None:
        for i, sample in enumerate(data_copy.loc[:,"Class"]):
            if sample != over_rep_class:
                data_copy.loc[i,"Class"] = -1
    
    full_tree = id3(data_copy, 'Class', max_depth)
    
    is_still_over_rep = True
        
    max_depth_to_trim = 0
    
    full_tree = sanitize_tree(full_tree)
    
    if max_class_samples is not None:
        classes = np.array(data_copy["Class"]).astype(int)
        classes = np.sort(classes)
        class_dict = {}
    
        for i in classes:
            class_dict[i] = class_dict.get(i, 0) + 1
        
        is_still_over_rep = False
        for class_name in class_dict:     
            if class_dict[class_name] > max_class_samples:
                is_still_over_rep = True
        
        if is_still_over_rep:
            max_depth_to_trim = max_depth_to_trim + 1
    else:
        return full_tree
    
    data_rows_removed = data_copy
    while is_still_over_rep:
        data_rows_removed = remove_data_rows_to_depth(full_tree, data_rows_removed, 1, max_depth_to_trim)
        
        is_still_over_rep = False
        for class_name in class_dict:     
            if class_dict[class_name] > max_class_samples:
                is_still_over_rep = True
        
        if is_still_over_rep:
            if max_depth_to_trim == max_depth:
                break
            max_depth_to_trim = max_depth_to_trim + 1
    
    trimmed_tree = trim_tree_at_depth(full_tree, 1, max_depth_to_trim)
    
    return trimmed_tree

def id3_estimate(tree, sample):
    
    id3_copy = tree
    while True:
        for key in id3_copy:
            var = key
        
        sample_val = sample[var]
        
        if sample_val not in id3_copy[var]:
            return -1

        tree_val = id3_copy[var][sample_val]
        
        if type(tree_val) is not dict:
            return tree_val
        else:
            id3_copy = tree_val