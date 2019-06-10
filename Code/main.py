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
data_A_normal = loadmat("../Data/My Data/Normal_0")
data_B_normal = loadmat("../Data/My Data/Normal_1")
data_C_normal = loadmat("../Data/My Data/Normal_2")
data_A_normal_DE = np.transpose(((data_A_normal['XC'])[0:102400]).reshape(50, 2048))
data_B_normal_DE = np.transpose(((data_B_normal['XC'])[0:102400]).reshape(50, 2048))
data_C_normal_DE = np.transpose(((data_C_normal['XC'])[0:102400]).reshape(50, 2048))
data_D_normal_DE = np.hstack([np.hstack([data_A_normal_DE, data_B_normal_DE]), data_C_normal_DE])


data_A_IR007 = loadmat("../Data/My Data/IR007_0")
data_B_IR007 = loadmat("../Data/My Data/IR007_1")
data_C_IR007 = loadmat("../Data/My Data/IR007_2")
data_A_IR007_DE = np.transpose(((data_A_IR007['XC'])[0:102400]).reshape(50, 2048))
data_B_IR007_DE = np.transpose(((data_B_IR007['XC'])[0:102400]).reshape(50, 2048))
data_C_IR007_DE = np.transpose(((data_C_IR007['XC'])[0:102400]).reshape(50, 2048))
data_D_IR007_DE = np.hstack([np.hstack([data_A_IR007_DE, data_B_IR007_DE]), data_C_IR007_DE])
data_A_B007 = loadmat("../Data/My Data/B007_0")
data_B_B007 = loadmat("../Data/My Data/B007_1")
data_C_B007 = loadmat("../Data/My Data/B007_2")
data_A_B007_DE = np.transpose(((data_A_B007['XC'])[0:102400]).reshape(50, 2048))
data_B_B007_DE = np.transpose(((data_B_B007['XC'])[0:102400]).reshape(50, 2048))
data_C_B007_DE = np.transpose(((data_C_B007['XC'])[0:102400]).reshape(50, 2048))
data_D_B007_DE = np.hstack([np.hstack([data_A_B007_DE, data_B_B007_DE]), data_C_B007_DE])
data_A_OR007 = loadmat("../Data/My Data/OR007@6_0")
data_B_OR007 = loadmat("../Data/My Data/OR007@6_1")
data_C_OR007 = loadmat("../Data/My Data/OR007@6_2")
data_A_OR007_DE = np.transpose(((data_A_OR007['XC'])[0:102400]).reshape(50, 2048))
data_B_OR007_DE = np.transpose(((data_B_OR007['XC'])[0:102400]).reshape(50, 2048))
data_C_OR007_DE = np.transpose(((data_C_OR007['XC'])[0:102400]).reshape(50, 2048))
data_D_OR007_DE = np.hstack([np.hstack([data_A_OR007_DE, data_B_OR007_DE]), data_C_OR007_DE])


data_A_IR014 = loadmat("../Data/My Data/IR014_0")
data_B_IR014 = loadmat("../Data/My Data/IR014_1")
data_C_IR014 = loadmat("../Data/My Data/IR014_2")
data_A_IR014_DE = np.transpose(((data_A_IR014['XC'])[0:102400]).reshape(50, 2048))
data_B_IR014_DE = np.transpose(((data_B_IR014['XC'])[0:102400]).reshape(50, 2048))
data_C_IR014_DE = np.transpose(((data_C_IR014['XC'])[0:102400]).reshape(50, 2048))
data_D_IR014_DE = np.hstack([np.hstack([data_A_IR014_DE, data_B_IR014_DE]), data_C_IR014_DE])
data_A_B014 = loadmat("../Data/My Data/B014_0")
data_B_B014 = loadmat("../Data/My Data/B014_1")
data_C_B014 = loadmat("../Data/My Data/B014_2")
data_A_B014_DE = np.transpose(((data_A_B014['XC'])[0:102400]).reshape(50, 2048))
data_B_B014_DE = np.transpose(((data_B_B014['XC'])[0:102400]).reshape(50, 2048))
data_C_B014_DE = np.transpose(((data_C_B014['XC'])[0:102400]).reshape(50, 2048))
data_D_B014_DE = np.hstack([np.hstack([data_A_B014_DE, data_B_B014_DE]), data_C_B014_DE])
data_A_OR014 = loadmat("../Data/My Data/OR014@6_0")
data_B_OR014 = loadmat("../Data/My Data/OR014@6_1")
data_C_OR014 = loadmat("../Data/My Data/OR014@6_2")
data_A_OR014_DE = np.transpose(((data_A_OR014['XC'])[0:102400]).reshape(50, 2048))
data_B_OR014_DE = np.transpose(((data_B_OR014['XC'])[0:102400]).reshape(50, 2048))
data_C_OR014_DE = np.transpose(((data_C_OR014['XC'])[0:102400]).reshape(50, 2048))
data_D_OR014_DE = np.hstack([np.hstack([data_A_OR014_DE, data_B_OR014_DE]), data_C_OR014_DE])


data_A_IR021 = loadmat("../Data/My Data/IR021_0")
data_B_IR021 = loadmat("../Data/My Data/IR021_1")
data_C_IR021 = loadmat("../Data/My Data/IR021_2")
data_A_IR021_DE = np.transpose(((data_A_IR021['XC'])[0:102400]).reshape(50, 2048))
data_B_IR021_DE = np.transpose(((data_B_IR021['XC'])[0:102400]).reshape(50, 2048))
data_C_IR021_DE = np.transpose(((data_C_IR021['XC'])[0:102400]).reshape(50, 2048))
data_D_IR021_DE = np.hstack([np.hstack([data_A_IR021_DE, data_B_IR021_DE]), data_C_IR021_DE])
data_A_B021 = loadmat("../Data/My Data/B021_0")
data_B_B021 = loadmat("../Data/My Data/B021_1")
data_C_B021 = loadmat("../Data/My Data/B021_2")
data_A_B021_DE = np.transpose(((data_A_B021['XC'])[0:102400]).reshape(50, 2048))
data_B_B021_DE = np.transpose(((data_B_B021['XC'])[0:102400]).reshape(50, 2048))
data_C_B021_DE = np.transpose(((data_C_B021['XC'])[0:102400]).reshape(50, 2048))
data_D_B021_DE = np.hstack([np.hstack([data_A_B021_DE, data_B_B021_DE]), data_C_B021_DE])
data_A_OR021 = loadmat("../Data/My Data/OR021@6_0")
data_B_OR021 = loadmat("../Data/My Data/OR021@6_1")
data_C_OR021 = loadmat("../Data/My Data/OR021@6_2")
data_A_OR021_DE = np.transpose(((data_A_OR021['XC'])[0:102400]).reshape(50, 2048))
data_B_OR021_DE = np.transpose(((data_B_OR021['XC'])[0:102400]).reshape(50, 2048))
data_C_OR021_DE = np.transpose(((data_C_OR021['XC'])[0:102400]).reshape(50, 2048))
data_D_OR021_DE = np.hstack([np.hstack([data_A_OR021_DE, data_B_OR021_DE]), data_C_OR021_DE])
print("Reading finished.")

min_max_scaler = MinMaxScaler()

data_A = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_A_normal_DE, data_A_IR007_DE]), data_A_B007_DE]), data_A_OR007_DE]), data_A_IR014_DE]), data_A_B014_DE]), data_A_OR014_DE]), data_A_IR021_DE]), data_A_B021_DE]), data_A_OR021_DE])
data_B = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_B_normal_DE, data_B_IR007_DE]), data_B_B007_DE]), data_B_OR007_DE]), data_B_IR014_DE]), data_B_B014_DE]), data_B_OR014_DE]), data_B_IR021_DE]), data_B_B021_DE]), data_B_OR021_DE])
data_C = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_C_normal_DE, data_C_IR007_DE]), data_C_B007_DE]), data_C_OR007_DE]), data_C_IR014_DE]), data_C_B014_DE]), data_C_OR014_DE]), data_C_IR021_DE]), data_C_B021_DE]), data_C_OR021_DE])
data_D = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_D_normal_DE, data_D_IR007_DE]), data_D_B007_DE]), data_D_OR007_DE]), data_D_IR014_DE]), data_D_B014_DE]), data_D_OR014_DE]), data_D_IR021_DE]), data_D_B021_DE]), data_D_OR021_DE])

n_feature = 18
t_A = np.zeros((n_feature, 500))
t_B = np.zeros((n_feature, 500))
t_C = np.zeros((n_feature, 500))
t_D = np.zeros((n_feature, 1500))

temp_A = data_A+abs(np.min(data_A))+1
temp_B = data_B+abs(np.min(data_B))+1
temp_C = data_C+abs(np.min(data_C))+1
temp_D = data_D+abs(np.min(data_D))+1

temp_A = np.abs(data_A)+1
temp_B = np.abs(data_B)+1
temp_C = np.abs(data_C)+1
temp_D = np.abs(data_D)+1

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

    mean = sum(data_B[:, i])/2048
    t_B[0, i] = max(abs(data_B[:, i]))-min(abs(data_B[:, i]))
    t_B[1, i] = max(abs(data_B[:, i]))
    t_B[2, i] = mean
    t_B[3, i] = sum(pow(data_B[:, i], 2))/2048
    t_B[4, i] = pow(sum(pow(data_B[:, i], 2))/2048, 0.5)
    t_B[5, i] = sum(abs(data_B[:, i]))/2048
    t_B[6, i] = pow(sum(pow(abs(data_B[:, i]), 0.5))/2048, 2)
    t_B[7, i] = sum(pow(data_B[:, i]-mean, 2))/2048
    t_B[8, i] = pow(sum(pow(data_B[:, i]-mean, 2))/2048, 0.5)
    t_B[9, i] = sum(pow(data_B[:, i], 3))/2048
    t_B[10, i] = sum(pow(data_B[:, i], 4))/2048
    t_B[11, i] = sum(pow(data_B[:, i], 3))/2048/pow(pow(sum(pow(data_B[:, i],2))/2048, 0.5), 3)
    t_B[12, i] = sum(pow(data_B[:, i], 4))/2048/pow(pow(sum(pow(data_B[:, i], 2))/2048, 0.5), 4)
    t_B[13, i] = max(abs(data_B[:, i]))/pow(sum(pow(data_B[:, i], 2))/2048, 0.5)
    t_B[14, i] = max(abs(data_B[:, i]))/(sum(abs(data_B[:, i]))/2048)
    t_B[15, i] = max(abs(data_B[:, i]))/pow(sum(pow(abs(data_B[:, i]),0.5))/2048, 2)
    t_B[16, i] = pow(sum(pow(data_B[:, i], 2))/2048, 0.5)/(sum(abs(data_B[:, i]))/2048)
    t_B[17, i] = np.mean(np.multiply(temp_B[:, i], np.log(temp_B[:, i])))

    mean = sum(data_C[:, i])/2048
    t_C[0, i] = max(abs(data_C[:, i]))-min(abs(data_C[:, i]))
    t_C[1, i] = max(abs(data_C[:, i]))
    t_C[2, i] = mean
    t_C[3, i] = sum(pow(data_C[:, i], 2))/2048
    t_C[4, i] = pow(sum(pow(data_C[:, i], 2))/2048, 0.5)
    t_C[5, i] = sum(abs(data_C[:, i]))/2048
    t_C[6, i] = pow(sum(pow(abs(data_C[:, i]), 0.5))/2048, 2)
    t_C[7, i] = sum(pow(data_C[:, i]-mean, 2))/2048
    t_C[8, i] = pow(sum(pow(data_C[:, i]-mean, 2))/2048, 0.5)
    t_C[9, i] = sum(pow(data_C[:, i], 3))/2048
    t_C[10, i] = sum(pow(data_C[:, i], 4))/2048
    t_C[11, i] = sum(pow(data_C[:, i], 3))/2048/pow(pow(sum(pow(data_C[:, i], 2))/2048, 0.5), 3)
    t_C[12, i] = sum(pow(data_C[:, i], 4))/2048/pow(pow(sum(pow(data_C[:, i], 2))/2048, 0.5), 4)
    t_C[13, i] = max(abs(data_C[:, i]))/pow(sum(pow(data_C[:, i], 2))/2048, 0.5)
    t_C[14, i] = max(abs(data_C[:, i]))/(sum(abs(data_C[:, i]))/2048)
    t_C[15, i] = max(abs(data_C[:, i]))/pow(sum(pow(abs(data_C[:, i]), 0.5))/2048, 2)
    t_C[16, i] = pow(sum(pow(data_C[:, i], 2))/2048, 0.5)/(sum(abs(data_C[:, i]))/2048)
    t_C[17, i] = np.mean(np.multiply(temp_C[:, i], np.log(temp_C[:, i])))

for i in range(1500):
    mean = sum(data_D[:, i])/2048
    t_D[0, i] = max(abs(data_D[:, i]))-min(abs(data_D[:, i]))
    t_D[1, i] = max(abs(data_D[:, i]))
    t_D[2, i] = mean
    t_D[3, i] = sum(pow(data_D[:, i], 2))/2048
    t_D[4, i] = pow(sum(pow(data_D[:, i], 2))/2048, 0.5)
    t_D[5, i] = sum(abs(data_D[:, i]))/2048
    t_D[6, i] = pow(sum(pow(abs(data_D[:, i]), 0.5))/2048, 2)
    t_D[7, i] = sum(pow(data_D[:, i]-mean, 2))/2048
    t_D[8, i] = pow(sum(pow(data_D[:, i]-mean, 2))/2048, 0.5)
    t_D[9, i] = sum(pow(data_D[:, i], 3))/2048
    t_D[10, i] = sum(pow(data_D[:, i], 4))/2048
    t_D[11, i] = sum(pow(data_D[:, i], 3))/2048/pow(pow(sum(pow(data_D[:, i], 2))/2048, 0.5), 3)
    t_D[12, i] = sum(pow(data_D[:, i], 4))/2048/pow(pow(sum(pow(data_D[:, i], 2))/2048, 0.5), 4)
    t_D[13, i] = max(abs(data_D[:, i]))/pow(sum(pow(data_D[:, i], 2))/2048, 0.5)
    t_D[14, i] = max(abs(data_D[:, i]))/(sum(abs(data_D[:, i]))/2048)
    t_D[15, i] = max(abs(data_D[:, i]))/pow(sum(pow(abs(data_D[:, i]), 0.5))/2048, 2)
    t_D[16, i] = pow(sum(pow(data_D[:, i], 2))/2048, 0.5)/(sum(abs(data_D[:, i]))/2048)
    t_D[17, i] = np.mean(np.multiply(temp_D[:, i], np.log(temp_D[:, i])))
print("Extraction finished.")

data_A_minmax = min_max_scaler.fit_transform(np.transpose(t_A))
data_B_minmax = min_max_scaler.fit_transform(np.transpose(t_B))
data_C_minmax = min_max_scaler.fit_transform(np.transpose(t_C))
data_D_minmax = min_max_scaler.fit_transform(np.transpose(t_D))

import warnings
warnings.filterwarnings("ignore")
plt.figure(1)
for k in range(n_feature):
    for i in range(4):
        plt.subplot(n_feature, 1, k+1)
        plt.xticks([])
        plt.yticks([])
        sns.distplot(t_A[k, (50*i):(50*(i+1))], rug=True, hist=False)
plt.show()

eval_A = np.zeros((500, 10))
eval_D = np.zeros((1500, 10))
for i in range(500):
    eval_A[i][(int)(i/50)] = 1
eval_B = eval_A
eval_C = eval_A
for i in range(1500):
    eval_D[i][(int)(i/150)] = 1

np.save('./data/data_A_minmax.npy', data_A_minmax)
np.save('./data/eval_A.npy', eval_A)
np.save('./data/data_B_minmax.npy', data_B_minmax)
np.save('./data/eval_B.npy', eval_B)
np.save('./data/data_C_minmax.npy', data_C_minmax)
np.save('./data/eval_C.npy', eval_C)
np.save('./data/data_D_minmax.npy', data_D_minmax)
np.save('./data/eval_D.npy', eval_D)
'''

data_A_minmax = np.load('./data/data_A_minmax.npy')
eval_A = np.load('./data/eval_A.npy')
data_B_minmax = np.load('./data/data_B_minmax.npy')
eval_B = np.load('./data/eval_B.npy')
data_C_minmax = np.load('./data/data_C_minmax.npy')
eval_C = np.load('./data/eval_C.npy')
data_D_minmax = np.load('./data/data_D_minmax.npy')
eval_D = np.load('./data/eval_D.npy')

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
    '''
    for i, size in enumerate(RBM_hidden_sizes):
        print('SSRBM: ', i, ' ', input_size, '->', size)
        rbm_list.append(SSRBM(input_size, size, 100, 0.3, 10, 0.5))
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
        X_ = np.transpose(np.matmul(np.sqrt(Sigma), np.transpose(U[0:inpX.shape[0], 0:inpX.shape[1]])))
        inpX = rbm.rbm_outpt(inpX, X_)

    nNet = NN(RBM_hidden_sizes, X_A_train, y_A_train, 1, 1000, 10)
    nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
    tr, te, result_tr, result_te = nNet.train(X_A_test, y_A_test)
    tra[epoch] = result_tr
    tea[epoch] = result_te
    '''
    #'''
    for i, size in enumerate(RBM_hidden_sizes):
        print('RBM: ', i, ' ', input_size, '->', size)
        rbm_list.append(RBM(input_size, size, 100, 0.3, 10))
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
np.save('./result/tr_DBN_sigmoid.npy', ave_tra)
np.save('./result/te_DBN_sigmoid.npy', ave_tea)
label = ['Training Dataset', 'Testing Dataset']
plt.plot(range(1500), ave_tra)
plt.plot(range(1500), ave_tea)
plt.legend(label)
plt.xlabel('Epoch')
plt.ylabel('Accuracy Rate')
plt.show()
'''
label = ['Training Dataset', 'Testing Dataset']
plt.plot(range(epoches), tra)
plt.plot(range(epoches), tea)
plt.legend(label)
plt.xlabel('Epoch')
plt.ylabel('Accuracy Rate')
plt.show()
print('Training Dataset Average: ', sum_tr/epoches)
print('Testing Dataset Average: ', sum_te/epoches)
'''