import numpy as np
import matplotlib.pyplot as plt
from random import randint

def generate_data(data):
    
    unique_data = np.unique(data, axis=0).astype(int) #remove duplicate rows
    #unique_data = np.array(data).astype(float)
    
    data_for_dist_cals = data
    
    classes = unique_data[:,0].astype(int)
    classes = np.sort(classes)
    
    class_dict = {}

    for i in classes:
        class_dict[i] = class_dict.get(i, 0) + 1
    
    greatest_class = max(class_dict, key=class_dict.get)
    greatest_num_samples = max(class_dict.values())
    
    # print(hist.keys())
    # print(hist.values())
    
    # a0 = plt.figure(0)
    # plt.bar(hist.keys(), hist.values())
    
    generated_data = []
    new_data_matrix = []
    
    for class_type in class_dict:
        if class_type == greatest_class:
            continue
        
        if class_dict.get(class_type) < 2:
            continue
        
        else:
            for sample_num, sample in enumerate(data_for_dist_cals):
                if int(sample[0]) == class_type:
                    new_data = []
                    new_data.extend([sample_num])
                    new_data.extend(sample[1:5])
                    
                    for hashed_sample_index in range(5):
                        range_to_hash = sample[hashed_sample_index*11 + 5:hashed_sample_index*11 + 16]
                        range_to_hash = tuple(range_to_hash)
                        range_hash = hash(range_to_hash)
                        new_data.extend([range_hash])
                        
                    new_data.extend(sample[60:62])
                    new_data_matrix.append(new_data)
                    pass

        data_vec_len = len(new_data_matrix[0])-1
        data_info = []
        
        for column in range(data_vec_len):
            
            data_val_dict = {}
            
            for row in new_data_matrix:
                data_val_dict[row[column+1]] = data_val_dict.get(row[column+1], 0) + 1
                
            data_info.append(data_val_dict)
        
        num_samples_in_class = class_dict.get(class_type)
        num_samples_to_generate = greatest_num_samples - num_samples_in_class
        
        for generated_sample_num in range(num_samples_to_generate):
        
            generated_data_row = []
            generated_data_row.extend([class_type])
        
            for var_num, var_dict in enumerate(data_info):
                rand_num = randint(1,num_samples_in_class)
                running_total = 0
                
                for var in var_dict:
                    running_total = var_dict.get(var) + running_total
                    
                    if rand_num <= running_total:
                        if var_num < 4 or var_num > 8:
                            generated_data_row.extend([var])
                        else:
                            for row in new_data_matrix:
                                if row[var_num+1] == var:
                                    data_to_append = data_for_dist_cals[row[0]][(var_num-4)*11 + 5:(var_num-4)*11 + 16]
                                    generated_data_row.extend(data_to_append)
                                    break
                        break
            
            generated_data.append(generated_data_row)
        
    return unique_data, np.array(generated_data).astype(int)