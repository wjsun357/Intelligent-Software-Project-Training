import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from DBN import NN, RBM
print("Reading data...")
data_A_normal = loadmat("../Data/Normal Baseline Data/Normal_0")
data_B_normal = loadmat("../Data/Normal Baseline Data/Normal_1")
data_C_normal = loadmat("../Data/Normal Baseline Data/Normal_2")
data_A_normal_DE = np.transpose(((data_A_normal['X097_DE_time'])[0:102400]).reshape(50,2048))
data_B_normal_DE = np.transpose(((data_B_normal['X098_DE_time'])[0:102400]).reshape(50,2048))
data_C_normal_DE = np.transpose(((data_C_normal['X099_DE_time'])[0:102400]).reshape(50,2048))
data_D_normal_DE = np.hstack([np.hstack([data_A_normal_DE,data_B_normal_DE]),data_C_normal_DE])


data_A_IR007 = loadmat("../Data/12k Drive End Bearing Fault Data/IR007_0")
data_B_IR007 = loadmat("../Data/12k Drive End Bearing Fault Data/IR007_1")
data_C_IR007 = loadmat("../Data/12k Drive End Bearing Fault Data/IR007_2")
data_A_IR007_DE = np.transpose(((data_A_IR007['X105_DE_time'])[0:102400]).reshape(50,2048))
data_B_IR007_DE = np.transpose(((data_B_IR007['X106_DE_time'])[0:102400]).reshape(50,2048))
data_C_IR007_DE = np.transpose(((data_C_IR007['X107_DE_time'])[0:102400]).reshape(50,2048))
data_D_IR007_DE = np.hstack([np.hstack([data_A_IR007_DE,data_B_IR007_DE]),data_C_IR007_DE])
data_A_B007 = loadmat("../Data/12k Drive End Bearing Fault Data/B007_0")
data_B_B007 = loadmat("../Data/12k Drive End Bearing Fault Data/B007_1")
data_C_B007 = loadmat("../Data/12k Drive End Bearing Fault Data/B007_2")
data_A_B007_DE = np.transpose(((data_A_B007['X118_DE_time'])[0:102400]).reshape(50,2048))
data_B_B007_DE = np.transpose(((data_B_B007['X119_DE_time'])[0:102400]).reshape(50,2048))
data_C_B007_DE = np.transpose(((data_C_B007['X120_DE_time'])[0:102400]).reshape(50,2048))
data_D_B007_DE = np.hstack([np.hstack([data_A_B007_DE,data_B_B007_DE]),data_C_B007_DE])
data_A_OR007 = loadmat("../Data/12k Drive End Bearing Fault Data/OR007@6_0")
data_B_OR007 = loadmat("../Data/12k Drive End Bearing Fault Data/OR007@6_1")
data_C_OR007 = loadmat("../Data/12k Drive End Bearing Fault Data/OR007@6_2")
data_A_OR007_DE = np.transpose(((data_A_OR007['X130_DE_time'])[0:102400]).reshape(50,2048))
data_B_OR007_DE = np.transpose(((data_B_OR007['X131_DE_time'])[0:102400]).reshape(50,2048))
data_C_OR007_DE = np.transpose(((data_C_OR007['X132_DE_time'])[0:102400]).reshape(50,2048))
data_D_OR007_DE = np.hstack([np.hstack([data_A_OR007_DE,data_B_OR007_DE]),data_C_OR007_DE])


data_A_IR014 = loadmat("../Data/12k Drive End Bearing Fault Data/IR014_0")
data_B_IR014 = loadmat("../Data/12k Drive End Bearing Fault Data/IR014_1")
data_C_IR014 = loadmat("../Data/12k Drive End Bearing Fault Data/IR014_2")
data_A_IR014_DE = np.transpose(((data_A_IR014['X169_DE_time'])[0:102400]).reshape(50,2048))
data_B_IR014_DE = np.transpose(((data_B_IR014['X170_DE_time'])[0:102400]).reshape(50,2048))
data_C_IR014_DE = np.transpose(((data_C_IR014['X171_DE_time'])[0:102400]).reshape(50,2048))
data_D_IR014_DE = np.hstack([np.hstack([data_A_IR014_DE,data_B_IR014_DE]),data_C_IR014_DE])
data_A_B014 = loadmat("../Data/12k Drive End Bearing Fault Data/B014_0")
data_B_B014 = loadmat("../Data/12k Drive End Bearing Fault Data/B014_1")
data_C_B014 = loadmat("../Data/12k Drive End Bearing Fault Data/B014_2")
data_A_B014_DE = np.transpose(((data_A_B014['X185_DE_time'])[0:102400]).reshape(50,2048))
data_B_B014_DE = np.transpose(((data_B_B014['X186_DE_time'])[0:102400]).reshape(50,2048))
data_C_B014_DE = np.transpose(((data_C_B014['X187_DE_time'])[0:102400]).reshape(50,2048))
data_D_B014_DE = np.hstack([np.hstack([data_A_B014_DE,data_B_B014_DE]),data_C_B014_DE])
data_A_OR014 = loadmat("../Data/12k Drive End Bearing Fault Data/OR014@6_0")
data_B_OR014 = loadmat("../Data/12k Drive End Bearing Fault Data/OR014@6_1")
data_C_OR014 = loadmat("../Data/12k Drive End Bearing Fault Data/OR014@6_2")
data_A_OR014_DE = np.transpose(((data_A_OR014['X197_DE_time'])[0:102400]).reshape(50,2048))
data_B_OR014_DE = np.transpose(((data_B_OR014['X198_DE_time'])[0:102400]).reshape(50,2048))
data_C_OR014_DE = np.transpose(((data_C_OR014['X199_DE_time'])[0:102400]).reshape(50,2048))
data_D_OR014_DE = np.hstack([np.hstack([data_A_OR014_DE,data_B_OR014_DE]),data_C_OR014_DE])


data_A_IR021 = loadmat("../Data/12k Drive End Bearing Fault Data/IR021_0")
data_B_IR021 = loadmat("../Data/12k Drive End Bearing Fault Data/IR021_1")
data_C_IR021 = loadmat("../Data/12k Drive End Bearing Fault Data/IR021_2")
data_A_IR021_DE = np.transpose(((data_A_IR021['X209_DE_time'])[0:102400]).reshape(50,2048))
data_B_IR021_DE = np.transpose(((data_B_IR021['X210_DE_time'])[0:102400]).reshape(50,2048))
data_C_IR021_DE = np.transpose(((data_C_IR021['X211_DE_time'])[0:102400]).reshape(50,2048))
data_D_IR021_DE = np.hstack([np.hstack([data_A_IR021_DE,data_B_IR021_DE]),data_C_IR021_DE])
data_A_B021 = loadmat("../Data/12k Drive End Bearing Fault Data/B021_0")
data_B_B021 = loadmat("../Data/12k Drive End Bearing Fault Data/B021_1")
data_C_B021 = loadmat("../Data/12k Drive End Bearing Fault Data/B021_2")
data_A_B021_DE = np.transpose(((data_A_B021['X222_DE_time'])[0:102400]).reshape(50,2048))
data_B_B021_DE = np.transpose(((data_B_B021['X223_DE_time'])[0:102400]).reshape(50,2048))
data_C_B021_DE = np.transpose(((data_C_B021['X224_DE_time'])[0:102400]).reshape(50,2048))
data_D_B021_DE = np.hstack([np.hstack([data_A_B021_DE,data_B_B021_DE]),data_C_B021_DE])
data_A_OR021 = loadmat("../Data/12k Drive End Bearing Fault Data/OR021@6_0")
data_B_OR021 = loadmat("../Data/12k Drive End Bearing Fault Data/OR021@6_1")
data_C_OR021 = loadmat("../Data/12k Drive End Bearing Fault Data/OR021@6_2")
data_A_OR021_DE = np.transpose(((data_A_OR021['X234_DE_time'])[0:102400]).reshape(50,2048))
data_B_OR021_DE = np.transpose(((data_B_OR021['X235_DE_time'])[0:102400]).reshape(50,2048))
data_C_OR021_DE = np.transpose(((data_C_OR021['X236_DE_time'])[0:102400]).reshape(50,2048))
data_D_OR021_DE = np.hstack([np.hstack([data_A_OR021_DE,data_B_OR021_DE]),data_C_OR021_DE])
print("Reading finished.")

min_max_scaler = MinMaxScaler()
'''
data_A_normal_DE_minmax = min_max_scaler.fit_transform(data_A_normal_DE)
data_B_normal_DE_minmax = min_max_scaler.fit_transform(data_B_normal_DE)
data_C_normal_DE_minmax = min_max_scaler.fit_transform(data_C_normal_DE)
data_D_normal_DE_minmax = min_max_scaler.fit_transform(data_D_normal_DE)

data_A_IR007_DE_minmax = min_max_scaler.fit_transform(data_A_IR007_DE)
data_B_IR007_DE_minmax = min_max_scaler.fit_transform(data_B_IR007_DE)
data_C_IR007_DE_minmax = min_max_scaler.fit_transform(data_C_IR007_DE)
data_D_IR007_DE_minmax = min_max_scaler.fit_transform(data_D_IR007_DE)
data_A_B007_DE_minmax = min_max_scaler.fit_transform(data_A_B007_DE)
data_B_B007_DE_minmax = min_max_scaler.fit_transform(data_B_B007_DE)
data_C_B007_DE_minmax = min_max_scaler.fit_transform(data_C_B007_DE)
data_D_B007_DE_minmax = min_max_scaler.fit_transform(data_D_B007_DE)
data_A_OR007_DE_minmax = min_max_scaler.fit_transform(data_A_OR007_DE)
data_B_OR007_DE_minmax = min_max_scaler.fit_transform(data_B_OR007_DE)
data_C_OR007_DE_minmax = min_max_scaler.fit_transform(data_C_OR007_DE)
data_D_OR007_DE_minmax = min_max_scaler.fit_transform(data_D_OR007_DE)

data_A_IR014_DE_minmax = min_max_scaler.fit_transform(data_A_IR014_DE)
data_B_IR014_DE_minmax = min_max_scaler.fit_transform(data_B_IR014_DE)
data_C_IR014_DE_minmax = min_max_scaler.fit_transform(data_C_IR014_DE)
data_D_IR014_DE_minmax = min_max_scaler.fit_transform(data_D_IR014_DE)
data_A_B014_DE_minmax = min_max_scaler.fit_transform(data_A_B014_DE)
data_B_B014_DE_minmax = min_max_scaler.fit_transform(data_B_B014_DE)
data_C_B014_DE_minmax = min_max_scaler.fit_transform(data_C_B014_DE)
data_D_B014_DE_minmax = min_max_scaler.fit_transform(data_D_B014_DE)
data_A_OR014_DE_minmax = min_max_scaler.fit_transform(data_A_OR014_DE)
data_B_OR014_DE_minmax = min_max_scaler.fit_transform(data_B_OR014_DE)
data_C_OR014_DE_minmax = min_max_scaler.fit_transform(data_C_OR014_DE)
data_D_OR014_DE_minmax = min_max_scaler.fit_transform(data_D_OR014_DE)

data_A_IR021_DE_minmax = min_max_scaler.fit_transform(data_A_IR021_DE)
data_B_IR021_DE_minmax = min_max_scaler.fit_transform(data_B_IR021_DE)
data_C_IR021_DE_minmax = min_max_scaler.fit_transform(data_C_IR021_DE)
data_D_IR021_DE_minmax = min_max_scaler.fit_transform(data_D_IR021_DE)
data_A_B021_DE_minmax = min_max_scaler.fit_transform(data_A_B021_DE)
data_B_B021_DE_minmax = min_max_scaler.fit_transform(data_B_B021_DE)
data_C_B021_DE_minmax = min_max_scaler.fit_transform(data_C_B021_DE)
data_D_B021_DE_minmax = min_max_scaler.fit_transform(data_D_B021_DE)
data_A_OR021_DE_minmax = min_max_scaler.fit_transform(data_A_OR021_DE)
data_B_OR021_DE_minmax = min_max_scaler.fit_transform(data_B_OR021_DE)
data_C_OR021_DE_minmax = min_max_scaler.fit_transform(data_C_OR021_DE)
data_D_OR021_DE_minmax = min_max_scaler.fit_transform(data_D_OR021_DE)
'''

data_A = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_A_normal_DE,data_A_IR007_DE]),data_A_B007_DE]),data_A_OR007_DE]),data_A_IR014_DE]),data_A_B014_DE]),data_A_OR014_DE]),data_A_IR021_DE]),data_A_B021_DE]),data_A_OR021_DE])
data_B = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_B_normal_DE,data_B_IR007_DE]),data_B_B007_DE]),data_B_OR007_DE]),data_B_IR014_DE]),data_B_B014_DE]),data_B_OR014_DE]),data_B_IR021_DE]),data_B_B021_DE]),data_B_OR021_DE])
data_C = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_C_normal_DE,data_C_IR007_DE]),data_C_B007_DE]),data_C_OR007_DE]),data_C_IR014_DE]),data_C_B014_DE]),data_C_OR014_DE]),data_C_IR021_DE]),data_C_B021_DE]),data_C_OR021_DE])
data_D = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_D_normal_DE,data_D_IR007_DE]),data_D_B007_DE]),data_D_OR007_DE]),data_D_IR014_DE]),data_D_B014_DE]),data_D_OR014_DE]),data_D_IR021_DE]),data_D_B021_DE]),data_D_OR021_DE])

data_A_minmax = np.transpose(min_max_scaler.fit_transform(data_A))
data_B_minmax = np.transpose(min_max_scaler.fit_transform(data_B))
data_C_minmax = np.transpose(min_max_scaler.fit_transform(data_C))
data_D_minmax = np.transpose(min_max_scaler.fit_transform(data_D))

eval_A = np.zeros((500,10))
eval_D = np.zeros((1500,10))
for i in range(500):
    eval_A[i][(int)(i/50)] = 1
eval_B = eval_A
eval_C = eval_A
for i in range(1500):
    eval_D[i][(int)(i/150)] = 1


X_A_train,X_A_test, y_A_train, y_A_test =train_test_split(data_A_minmax,eval_A,test_size=0.4, random_state=0)
RBM_hidden_sizes = [1200, 600, 300]
inpX = X_A_train
rbm_list = []
input_size = inpX.shape[1]
for i, size in enumerate(RBM_hidden_sizes):
    print('RBM: ', i, ' ', input_size, '->', size)
    rbm_list.append(RBM(input_size, size))
    input_size = size

for rbm in rbm_list:
    print('New RBM:')
    rbm.train(inpX)
    inpX = rbm.rbm_outpt(inpX)

nNet = NN(RBM_hidden_sizes, X_A_train, y_A_train)
nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
nNet.train()