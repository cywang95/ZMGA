import torch
from sklearn.preprocessing import MinMaxScaler
import h5py
import warnings
import numpy as np
from utils import X2A

warnings.filterwarnings("ignore")
ALL_data = dict( Dataset_6 = {1: 'SNARE', 2: 'SNARE', 'N': 1047, 'K': 4, 'V': 2, 'n_input': [500,7136], 'n_hid': [10,256], 'n_output': 64})
path = '/mnt/d/Code/ZMGA/data/'
def load_data(dataset):
    data = h5py.File(path + dataset[1] + ".mat")
    X = []
    Y = []
    A = []
    Label = np.array(data['Y']).T
    Label = Label.reshape(Label.shape[0])
    mm = MinMaxScaler()
    for i in range(data['X'].shape[1]):
        diff_view = data[data['X'][0, i]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        std_view = mm.fit_transform(diff_view)
        X.append(std_view)
        Y.append(Label)
    size = len(Y[0])
    view_num = len(X)
    index = [i for i in range(size)]
    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]
        A.append(X2A(X[v], 'p'))
    for v in range(view_num):
        X[v] = torch.from_numpy(X[v])
    return X, Y, A





