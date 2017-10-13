from __future__ import division
import numpy as np
import os
from pprint import pprint 
from copy import copy, deepcopy

def read_file(filepath):
    data = []
    with open(filepath) as f:
        for (idx,line) in enumerate(f):
            if idx < 3:
                continue
            temp = line.strip().split(' ')
            temp = [float(t) for t in temp]
            data.append(temp)
    return data # ignore first three lines

def read_all_data():
    data_dir_local = 'data'
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, data_dir_local)
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    all_data = []
    for file in files:
        # print file 
        data_curr = read_file(file)
        all_data = all_data + data_curr
    return all_data

def svm_train(train_data, 
        n_time_steps,
        class_idx,  
        lambda_factor):
    theta = np.zeros((1, 10))
    weights = np.zeros((n_time_steps, 10), dtype=float)
    # binary_data = deepcopy(train_data)
    # print binary_data[:, 4]==class_idx
    # binary_data[ binary_data[:, 4]==class_idx ] = 1 
    # binary_data[ binary_data[:, 4]!=class_idx ] = -1 
    for time_step in range(n_time_steps-1):
        weights[time_step,:] = 1/float(lambda_factor*(time_step+1))*(theta)
        idx = np.random.random_integers(len(train_data)-1)
        y_i = train_data[idx, 4] # label
        if y_i == class_idx:
            y_i = 1
        else:
            y_i = -1
        x_i = train_data[idx, 5:] # featss
        if y_i*np.dot(weights[time_step,:], x_i) < 1:
            theta = theta + y_i*x_i
    w_optimal = 1/float(n_time_steps)*(np.sum(weights, axis=0))
    return w_optimal

def svm_test(test_data, w_optimal_dict, class_id_map):
    n_correct = 0
    for idx in range(len(test_data)):
        y_i = test_data[idx, 4] # label
        curr_score = {}
        for svm_class_name, svm_class_idx in class_id_map.iteritems():
            w_optimal = w_optimal_dict[svm_class_name]
            if y_i == svm_class_idx:
                y_i_test = 1
            else:
                y_i_test = -1
            curr_score[svm_class_idx] = np.dot(w_optimal, test_data[idx, 5:])
        
        pred_idx = max(curr_score, key=curr_score.get)
        # print pred_idx
        if pred_idx == y_i:
            n_correct += 1
    print "n_correct {}, len(test_data) {}".format(n_correct, len(test_data))
    print "accuracy  percentage {} %".format(100*n_correct/len(test_data))

# def plot_confusion(results):

def main():
    np.random.seed(324)
    data_as_list = read_all_data()
    data_arr = np.array(data_as_list)
    # shuffle
    np.random.shuffle(data_arr)
    n_train_samples = int(0.8*len(data_arr))
    train_data = data_arr[:n_train_samples]
    test_data = data_arr[n_train_samples:]
    # print "len(train_data) {}, len(test_data) {}".format(len(train_data), len(test_data))
 
    # format: x y z node_id node_label [features]
    # train_features = train_data[:, 5:]

    class_id_map = {'Veg' : 1004, 'Wire' : 1100, 'Pole' : 1103, 'Ground' : 1200, 'Facade' : 1400}
    # id_class_map = {1004:'Veg', 1100:'Wire', 1103:'Pole', 1200:'Ground', 1400:'Facade'}

    n_time_steps = len(train_data)
    class_idx = class_id_map['Veg']
    lambda_factor = 1

    w_optimal = {}
    w_optimal['Veg'] = svm_train(train_data, n_time_steps, class_id_map['Veg'], lambda_factor)
    w_optimal['Wire'] = svm_train(train_data, n_time_steps, class_id_map['Wire'], lambda_factor)
    w_optimal['Pole'] = svm_train(train_data, n_time_steps, class_id_map['Pole'], lambda_factor)
    w_optimal['Ground'] = svm_train(train_data, n_time_steps, class_id_map['Ground'], lambda_factor)
    w_optimal['Facade'] = svm_train(train_data, n_time_steps, class_id_map['Facade'], lambda_factor)

    svm_test(test_data, w_optimal, class_id_map)

if __name__=='__main__':
    main()