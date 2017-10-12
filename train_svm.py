import numpy as np
import os
from pprint import pprint 

def read_file(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(line.strip().split(' '))
    return data[3:] # ignore first three lines

def read_data():
    data_dir_local = 'data'
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, data_dir_local)
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    all_data = []
    for file in files:
        print file 
        data_curr = read_file(file)
        all_data = all_data + data_curr
    return all_data

def main():
    data_as_list = read_data()
    data_arr = np.array(data_as_list)
    # print data_arr.shape
    # format: x y z node_id node_label [features]
    features = data_arr[:, 5:]
    # print features.shape 

if __name__=='__main__':
    main()