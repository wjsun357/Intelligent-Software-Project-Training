import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from DBN import NN, RBM
#from SSDBN import NN, SSRBM

print("Reading data...")
data_A_normal = loadmat("../Data/Normal Baseline Data/Normal_0")
data_A_normal_DE = np.transpose(((data_A_normal['X097_DE_time'])[0:102400]).reshape(50, 2048))


data_A_IR007 = loadmat("../Data/12k Drive End Bearing Fault Data/IR007_0")
data_A_IR007_DE = np.transpose(((data_A_IR007['X105_DE_time'])[0:102400]).reshape(50, 2048))
data_A_B007 = loadmat("../Data/12k Drive End Bearing Fault Data/B007_0")
data_A_B007_DE = np.transpose(((data_A_B007['X118_DE_time'])[0:102400]).reshape(50, 2048))
data_A_OR007 = loadmat("../Data/12k Drive End Bearing Fault Data/OR007@6_0")
data_A_OR007_DE = np.transpose(((data_A_OR007['X130_DE_time'])[0:102400]).reshape(50, 2048))


data_A_IR014 = loadmat("../Data/12k Drive End Bearing Fault Data/IR014_0")
data_A_IR014_DE = np.transpose(((data_A_IR014['X169_DE_time'])[0:102400]).reshape(50, 2048))
data_A_B014 = loadmat("../Data/12k Drive End Bearing Fault Data/B014_0")
data_A_B014_DE = np.transpose(((data_A_B014['X185_DE_time'])[0:102400]).reshape(50, 2048))
data_A_OR014 = loadmat("../Data/12k Drive End Bearing Fault Data/OR014@6_0")
data_A_OR014_DE = np.transpose(((data_A_OR014['X197_DE_time'])[0:102400]).reshape(50, 2048))


data_A_IR021 = loadmat("../Data/12k Drive End Bearing Fault Data/IR021_0")
data_A_IR021_DE = np.transpose(((data_A_IR021['X209_DE_time'])[0:102400]).reshape(50, 2048))
data_A_B021 = loadmat("../Data/12k Drive End Bearing Fault Data/B021_0")
data_A_B021_DE = np.transpose(((data_A_B021['X222_DE_time'])[0:102400]).reshape(50, 2048))
data_A_OR021 = loadmat("../Data/12k Drive End Bearing Fault Data/OR021@6_0")
data_A_OR021_DE = np.transpose(((data_A_OR021['X234_DE_time'])[0:102400]).reshape(50, 2048))
print("Reading finished.")

min_max_scaler = MinMaxScaler()

data_A = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_A_normal_DE, data_A_IR007_DE]), data_A_B007_DE]), data_A_OR007_DE]), data_A_IR014_DE]), data_A_B014_DE]), data_A_OR014_DE]), data_A_IR021_DE]), data_A_B021_DE]), data_A_OR021_DE])

n_feature = 18
t_A = np.zeros((n_feature, 500))

temp_A = data_A+abs(np.min(data_A))+1

temp_A = np.abs(data_A)+1

for i in range(500):
    mean = sum(data_A[:, i])/2048
    t_A[0, i] = max(abs(data_A[:, i]))-min(abs(data_A[:, i]))
    t_A[1, i] = max(abs(data_A[:, i]))
    t_A[2, i] = mean
    t_A[3, i] = sum(pow(data_A[:, i], 2))/2048
    t_A[4, i] = pow(sum(pow(data_A[:, i], 2))/2048, 0.5)
    t_A[5, i] = sum(abs(data_A[:, i]))/2048
    t_A[6, i] = pow(sum(pow(abs(data_A[:, i]), 0.5))/2048, 2)
    t_A[7, i] = sum(pow(data_A[:, i]-mean, 2))/2048
    t_A[8, i] = pow(sum(pow(data_A[:, i]-mean, 2))/2048, 0.5)
    t_A[9, i] = sum(pow(data_A[:, i], 3))/2048
    t_A[10, i] = sum(pow(data_A[:, i], 4))/2048
    t_A[11, i] = sum(pow(data_A[:, i], 3))/2048/pow(pow(sum(pow(data_A[:, i], 2))/2048, 0.5), 3)
    t_A[12, i] = sum(pow(data_A[:, i], 4))/2048/pow(pow(sum(pow(data_A[:, i], 2))/2048, 0.5), 4)
    t_A[13, i] = max(abs(data_A[:, i]))/pow(sum(pow(data_A[:, i], 2))/2048, 0.5)
    t_A[14, i] = max(abs(data_A[:, i]))/(sum(abs(data_A[:, i]))/2048)
    t_A[15, i] = max(abs(data_A[:, i]))/pow(sum(pow(abs(data_A[:, i]), 0.5))/2048, 2)
    t_A[16, i] = pow(sum(pow(data_A[:, i], 2))/2048, 0.5)/(sum(abs(data_A[:, i]))/2048)
    t_A[17, i] = np.mean(np.multiply(temp_A[:, i], np.log(temp_A[:, i])))
print("Extraction finished.")

data_A_minmax = min_max_scaler.fit_transform(np.transpose(t_A))

eval_A = np.zeros((500, 10))
for i in range(500):
    eval_A[i][(int)(i/50)] = 1

np.save('./new_data/original_data_A_minmax.npy', data_A_minmax)
np.save('./new_data/eval_A.npy', eval_A)

epoches = 10
tra = np.zeros((epoches, 1500))
tea = np.zeros((epoches, 1500))
sum_tr = 0
sum_te = 0
for epoch in range(epoches):
    X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(data_A_minmax[:, 0:18], eval_A, test_size=0.4, random_state=0)
    RBM_hidden_sizes = [15, 13, 10]
    inpX = X_A_train
    rbm_list = []
    input_size = inpX.shape[1]
    #'''
    for i, size in enumerate(RBM_hidden_sizes):
        print('RBM: ', i, ' ', input_size, '->', size)
        rbm_list.append(RBM(input_size, size, 1, 0, 10))
        input_size = size

    for rbm in rbm_list:
        print('New RBM:')
        rbm.train(inpX)
        inpX = rbm.rbm_outpt(inpX)

    nNet = NN(RBM_hidden_sizes, X_A_train, y_A_train, 1, 1500, 10)
    nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
    tr, te, result_tr, result_te = nNet.train(X_A_test, y_A_test)
    tra[epoch] = result_tr
    tea[epoch] = result_te
    #'''
ave_tra = []
ave_tea = []
for i in range(1500):
    Sum_tra = 0
    Sum_tea = 0
    for j in range(epoches):
        Sum_tra = Sum_tra + tra[j][i]
        Sum_tea = Sum_tea + tea[j][i]
    ave_tra.append(Sum_tra / epoches)
    ave_tea.append(Sum_tea / epoches)
np.save('./result/tr_original_sigmoid.npy', ave_tra)
np.save('./result/te_original_sigmoid.npy', ave_tea)
label = ['Training Dataset', 'Testing Dataset']
plt.plot(range(1500), ave_tra)
plt.plot(range(1500), ave_tea)
plt.legend(label)
plt.xlabel('Epoch')
plt.ylabel('Accuracy Rate')
plt.show()