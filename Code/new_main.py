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
'''
print("Reading data...")
data_A_normal = loadmat("../Data/New Data/97_n")
data_A_normal_DE = np.transpose(((data_A_normal['X097_DE_time'])[0:204800]).reshape(100,2048))
data_A_IR007 = loadmat("../Data/New Data/105_n")
data_A_IR007_DE = np.transpose(((data_A_IR007['X105_DE_time'])[0:102400]).reshape(50,2048))
data_A_B007 = loadmat("../Data/New Data/118_n")
data_A_B007_DE = np.transpose(((data_A_B007['X118_DE_time'])[0:102400]).reshape(50,2048))
data_A_OR007 = loadmat("../Data/New Data/130_n")
data_A_OR007_DE = np.transpose(((data_A_OR007['X130_DE_time'])[0:102400]).reshape(50,2048))
print("Reading finished.")
min_max_scaler = MinMaxScaler()
data_A = np.hstack([np.hstack([np.hstack([data_A_normal_DE,data_A_IR007_DE]),data_A_B007_DE]),data_A_OR007_DE])

t_A = np.zeros((17,250))
for i in range(250):
    mean = sum(data_A[:,i])/2048
    t_A[0,i] = max(abs(data_A[:,i]))-min(abs(data_A[:,i]))
    t_A[1,i] = max(abs(data_A[:,i]))
    t_A[2,i] = mean
    t_A[3,i] = sum(pow(data_A[:,i],2))/2048
    t_A[4,i] = pow(sum(pow(data_A[:,i],2))/2048,0.5)
    t_A[5,i] = sum(abs(data_A[:,i]))/2048
    t_A[6,i] = pow(sum(pow(abs(data_A[:,i]),0.5))/2048,2)
    t_A[7,i] = sum(pow(data_A[:,i]-mean,2))/2048
    t_A[8,i] = pow(sum(pow(data_A[:,i]-mean,2))/2048,0.5)
    t_A[9,i] = sum(pow(data_A[:,i],3))/2048
    t_A[10,i] = sum(pow(data_A[:,i],4))/2048
    t_A[11,i] = sum(pow(data_A[:,i],3))/2048/pow(pow(sum(pow(data_A[:,i],2))/2048,0.5),3)
    t_A[12,i] = sum(pow(data_A[:,i],4))/2048/pow(pow(sum(pow(data_A[:,i],2))/2048,0.5),4)
    t_A[13,i] = max(abs(data_A[:,i]))/pow(sum(pow(data_A[:,i],2))/2048,0.5)
    t_A[14,i] = max(abs(data_A[:,i]))/(sum(abs(data_A[:,i]))/2048)
    t_A[15,i] = max(abs(data_A[:,i]))/pow(sum(pow(abs(data_A[:,i]),0.5))/2048,2)
    t_A[16,i] = pow(sum(pow(data_A[:,i],2))/2048,0.5)/(sum(abs(data_A[:,i]))/2048)
print("Extraction finished.")
'''
'''
import warnings
warnings.filterwarnings("ignore")
n_feature = 17
plt.figure(1)
for k in range(n_feature):
    for i in range(4):
        plt.subplot(n_feature,1,k+1)
        sns.distplot(t_A[k,(50*i):(50*(i+1))], rug=True, hist=False)
plt.show()
'''

'''
data_A_minmax = min_max_scaler.fit_transform(np.transpose(t_A))

eval_A = np.zeros((250,4))
for i in range(100):
    eval_A[i][(int)(i/100)] = 1
for i in range(100,250):
    eval_A[i][(int)(i/50-1)] = 1

np.save('./new_data/data_A_minmax.npy',data_A_minmax)
np.save('./new_data/eval_A.npy',eval_A)
'''


data_A_minmax = np.load('./data/data_A_minmax.npy')
eval_A = np.load('./data/eval_A.npy')

X_A_train, X_A_test, y_A_train, y_A_test = train_test_split(data_A_minmax, eval_A, test_size=0.4, random_state=0)
RBM_hidden_sizes = [15, 13, 10]
inpX = X_A_train
rbm_list = []
input_size = inpX.shape[1]
#'''
for i, size in enumerate(RBM_hidden_sizes):
    print('RBM: ', i, ' ', input_size, '->', size)
    rbm_list.append(RBM(input_size, size, 50, 0.01, 3))
    input_size = size

for rbm in rbm_list:
    print('New RBM:')
    rbm.train(inpX)
    inpX = rbm.rbm_outpt(inpX)

nNet = NN(RBM_hidden_sizes, X_A_train, y_A_train, 1, 500, 3)
nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
nNet.train(X_A_test, y_A_test)
#'''
'''
for i, size in enumerate(RBM_hidden_sizes):
    print('SSRBM: ', i, ' ', input_size, '->', size)
    rbm_list.append(SSRBM(input_size, size, 100, 0.001, 3, 0.5))
    input_size = size

S = np.zeros([X_A_train.shape[0], X_A_train.shape[0]])
for i in range(X_A_train.shape[0]):
    for j in range(X_A_train.shape[0]):
        if y_A_train[i].all() == y_A_train[j].all():
            S[i, j] = 0
        else:
            S[i, j] = np.matmul(X_A_train[i], np.transpose(X_A_train[j])) / (np.power(np.sum(np.power(X_A_train[i], 2)), 0.5) * np.power(np.sum(np.power(X_A_train[j], 2)), 0.5))

for rbm in rbm_list:
    print('New SSRBM:')
    rbm.train(inpX, S)
    U, sigma, VT = np.linalg.svd(S)
    Sigma = np.zeros([inpX.shape[1], inpX.shape[1]])
    for i in range(inpX.shape[1]):
        Sigma[i, i] = sigma[i]
    X_ = np.transpose(np.matmul(Sigma, np.transpose(U[0:inpX.shape[0], 0:inpX.shape[1]])))
    inpX = rbm.rbm_outpt(inpX, X_)

nNet = NN(RBM_hidden_sizes, X_A_train, y_A_train, 1, 0.5, 1000, 3)
nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
nNet.train(X_A_test, y_A_test)
'''