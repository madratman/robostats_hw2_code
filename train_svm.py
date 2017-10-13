from __future__ import division
import numpy as np
import os
from pprint import pprint 
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import time

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
        weights[time_step,:] = theta / (lambda_factor*(time_step+1))
        idx = np.random.random_integers(len(train_data)-1)
        y_i = train_data[idx, 4] # label
        if y_i == class_idx:
            y_i = 1
        else:
            y_i = -1
        x_i = train_data[idx, 5:] # featss
        if y_i*np.dot(weights[time_step,:], x_i) < 1:
            theta = theta + y_i*x_i
    w_optimal = 1/n_time_steps*np.sum(weights, axis=0)
    return w_optimal

def svm_test(test_data, w_optimal_dict, class_id_map):
    n_correct = 0
    confusion_matrix = np.zeros((5,5))
    class_indices_list = [1004, 1100, 1103, 1200, 1400] # such shitty code, much wow

    for idx in range(len(test_data)):
        y_i = test_data[idx, 4] # label
        curr_score = {}
        for svm_class_name, svm_class_idx in class_id_map.iteritems():
            w_optimal = w_optimal_dict[svm_class_name]
            # print svm_class_name, w_optimal
            if y_i == svm_class_idx:
                y_i_test = 1
            else:
                y_i_test = -1
            curr_score[svm_class_idx] = np.dot(w_optimal, test_data[idx, 5:])
        
        pred_idx = max(curr_score, key=curr_score.get)
        # print curr_score
        # print pred_idx, y_i
        confusion_matrix[class_indices_list.index(y_i), class_indices_list.index(pred_idx)] += 1
        # print pred_idx
        if pred_idx == y_i:
            n_correct += 1
    print "num correctly classified{}, total number of test samples {} \n".format(n_correct, len(test_data))
    print "Classification Accuracy {} % \n".format(100*n_correct/len(test_data))
    labels = class_id_map.keys()

    # normalize confusion_matrix
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    print "Confusion matrix\n"
    print confusion_matrix
    np.savetxt("confusion_matrix.csv", confusion_matrix, delimiter=",")
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    np.savetxt("norm_confusion_matrix.csv", confusion_matrix, delimiter=",")
    np.set_printoptions(precision=3)
    print "\nNormalized Confusion matrix\n"
    print confusion_matrix
    plt.imshow(confusion_matrix, cmap='jet', interpolation='nearest')
    plt.show()

def main():
    # np.random.seed(324)
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
    # n_time_steps = 10000
    # class_idx = class_id_map['Veg']
    lambda_factor = 0.01

    train_times = {}
    w_optimal = {}
    start_time = time.time()
    w_optimal['Veg'] = svm_train(train_data, n_time_steps, class_id_map['Veg'], lambda_factor)
    train_times['Veg'] = time.time() - start_time

    start_time = time.time()
    w_optimal['Wire'] = svm_train(train_data, n_time_steps, class_id_map['Wire'], lambda_factor)
    train_times['Wire'] = time.time() - start_time
    
    start_time = time.time()
    w_optimal['Pole'] = svm_train(train_data, n_time_steps, class_id_map['Pole'], lambda_factor)
    train_times['Pole'] = time.time() - start_time
    
    start_time = time.time()
    w_optimal['Ground'] = svm_train(train_data, n_time_steps, class_id_map['Ground'], lambda_factor)
    train_times['Ground'] = time.time() - start_time
    
    start_time = time.time()
    w_optimal['Facade'] = svm_train(train_data, n_time_steps, class_id_map['Facade'], lambda_factor)
    train_times['Facade'] = time.time() - start_time


    start_time = time.time()
    svm_test(test_data, w_optimal, class_id_map)
    test_time = time.time() - start_time

    print "\n Lambda {}".format(lambda_factor)
    print "\n Number of training samples {}".format(n_time_steps)
    print "Training times per SVM"
    pprint(train_times)
    print "Total Training Time {}".format(sum(train_times.values()))

    print "Number Of Testing Samples {}".format(len(test_data))
    print "Test time {}".format(test_time)


if __name__=='__main__':
    main()

