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
data_A_normal_reduced = loadmat("../Data/My Data/Normal_0")
data_B_normal_reduced = loadmat("../Data/My Data/Normal_1")
data_C_normal_reduced = loadmat("../Data/My Data/Normal_2")
data_A_normal_original = loadmat("../Data/Normal Baseline Data/Normal_0")
data_B_normal_original = loadmat("../Data/Normal Baseline Data/Normal_1")
data_C_normal_original = loadmat("../Data/Normal Baseline Data/Normal_2")
data_A_normal_reduced_DE = np.transpose(((data_A_normal_reduced['XC'])[0:102400]).reshape(50, 2048))
data_B_normal_reduced_DE = np.transpose(((data_B_normal_reduced['XC'])[0:102400]).reshape(50, 2048))
data_C_normal_reduced_DE = np.transpose(((data_C_normal_reduced['XC'])[0:102400]).reshape(50, 2048))
data_A_normal_original_DE = np.transpose(((data_A_normal_original['X097_DE_time'])[0:102400]).reshape(50, 2048))
data_B_normal_original_DE = np.transpose(((data_B_normal_original['X098_DE_time'])[0:102400]).reshape(50, 2048))
data_C_normal_original_DE = np.transpose(((data_C_normal_original['X099_DE_time'])[0:102400]).reshape(50, 2048))
data_D_normal_reduced_DE = np.hstack([np.hstack([data_A_normal_reduced_DE, data_B_normal_reduced_DE]), data_C_normal_reduced_DE])
data_D_normal_original_DE = np.hstack([np.hstack([data_A_normal_original_DE, data_B_normal_original_DE]), data_C_normal_original_DE])


data_A_IR007_reduced = loadmat("../Data/My Data/IR007_0")
data_B_IR007_reduced = loadmat("../Data/My Data/IR007_1")
data_C_IR007_reduced = loadmat("../Data/My Data/IR007_2")
data_A_IR007_original = loadmat("../Data/12k Drive End Bearing Fault Data/IR007_0")
data_B_IR007_original = loadmat("../Data/12k Drive End Bearing Fault Data/IR007_1")
data_C_IR007_original = loadmat("../Data/12k Drive End Bearing Fault Data/IR007_2")
data_A_IR007_reduced_DE = np.transpose(((data_A_IR007_reduced['XC'])[0:102400]).reshape(50, 2048))
data_B_IR007_reduced_DE = np.transpose(((data_B_IR007_reduced['XC'])[0:102400]).reshape(50, 2048))
data_C_IR007_reduced_DE = np.transpose(((data_C_IR007_reduced['XC'])[0:102400]).reshape(50, 2048))
data_A_IR007_original_DE = np.transpose(((data_A_IR007_original['X105_DE_time'])[0:102400]).reshape(50, 2048))
data_B_IR007_original_DE = np.transpose(((data_B_IR007_original['X106_DE_time'])[0:102400]).reshape(50, 2048))
data_C_IR007_original_DE = np.transpose(((data_C_IR007_original['X107_DE_time'])[0:102400]).reshape(50, 2048))
data_D_IR007_reduced_DE = np.hstack([np.hstack([data_A_IR007_reduced_DE, data_B_IR007_reduced_DE]), data_C_IR007_reduced_DE])
data_D_IR007_original_DE = np.hstack([np.hstack([data_A_IR007_original_DE, data_B_IR007_original_DE]), data_C_IR007_original_DE])

data_A_B007_reduced = loadmat("../Data/My Data/B007_0")
data_B_B007_reduced = loadmat("../Data/My Data/B007_1")
data_C_B007_reduced = loadmat("../Data/My Data/B007_2")
data_A_B007_original = loadmat("../Data/12k Drive End Bearing Fault Data/B007_0")
data_B_B007_original = loadmat("../Data/12k Drive End Bearing Fault Data/B007_1")
data_C_B007_original = loadmat("../Data/12k Drive End Bearing Fault Data/B007_2")
data_A_B007_reduced_DE = np.transpose(((data_A_B007_reduced['XC'])[0:102400]).reshape(50, 2048))
data_B_B007_reduced_DE = np.transpose(((data_B_B007_reduced['XC'])[0:102400]).reshape(50, 2048))
data_C_B007_reduced_DE = np.transpose(((data_C_B007_reduced['XC'])[0:102400]).reshape(50, 2048))
data_A_B007_original_DE = np.transpose(((data_A_B007_original['X118_DE_time'])[0:102400]).reshape(50, 2048))
data_B_B007_original_DE = np.transpose(((data_B_B007_original['X119_DE_time'])[0:102400]).reshape(50, 2048))
data_C_B007_original_DE = np.transpose(((data_C_B007_original['X120_DE_time'])[0:102400]).reshape(50, 2048))
data_D_B007_reduced_DE = np.hstack([np.hstack([data_A_B007_reduced_DE, data_B_B007_reduced_DE]), data_C_B007_reduced_DE])
data_D_B007_original_DE = np.hstack([np.hstack([data_A_B007_original_DE, data_B_B007_original_DE]), data_C_B007_original_DE])


data_A_OR007_reduced = loadmat("../Data/My Data/OR007@6_0")
data_B_OR007_reduced = loadmat("../Data/My Data/OR007@6_1")
data_C_OR007_reduced = loadmat("../Data/My Data/OR007@6_2")
data_A_OR007_original = loadmat("../Data/12k Drive End Bearing Fault Data/OR007@6_0")
data_B_OR007_original = loadmat("../Data/12k Drive End Bearing Fault Data/OR007@6_1")
data_C_OR007_original = loadmat("../Data/12k Drive End Bearing Fault Data/OR007@6_2")
data_A_OR007_reduced_DE = np.transpose(((data_A_OR007_reduced['XC'])[0:102400]).reshape(50, 2048))
data_B_OR007_reduced_DE = np.transpose(((data_B_OR007_reduced['XC'])[0:102400]).reshape(50, 2048))
data_C_OR007_reduced_DE = np.transpose(((data_C_OR007_reduced['XC'])[0:102400]).reshape(50, 2048))
data_A_OR007_original_DE = np.transpose(((data_A_OR007_original['X130_DE_time'])[0:102400]).reshape(50, 2048))
data_B_OR007_original_DE = np.transpose(((data_B_OR007_original['X131_DE_time'])[0:102400]).reshape(50, 2048))
data_C_OR007_original_DE = np.transpose(((data_C_OR007_original['X132_DE_time'])[0:102400]).reshape(50, 2048))
data_D_OR007_reduced_DE = np.hstack([np.hstack([data_A_OR007_reduced_DE, data_B_OR007_reduced_DE]), data_C_OR007_reduced_DE])
data_D_OR007_original_DE = np.hstack([np.hstack([data_A_OR007_original_DE, data_B_OR007_original_DE]), data_C_OR007_original_DE])


data_A_IR014_reduced = loadmat("../Data/My Data/IR014_0")
data_B_IR014_reduced = loadmat("../Data/My Data/IR014_1")
data_C_IR014_reduced = loadmat("../Data/My Data/IR014_2")
data_A_IR014_original = loadmat("../Data/12k Drive End Bearing Fault Data/IR014_0")
data_B_IR014_original = loadmat("../Data/12k Drive End Bearing Fault Data/IR014_1")
data_C_IR014_original = loadmat("../Data/12k Drive End Bearing Fault Data/IR014_2")
data_A_IR014_reduced_DE = np.transpose(((data_A_IR014_reduced['XC'])[0:102400]).reshape(50, 2048))
data_B_IR014_reduced_DE = np.transpose(((data_B_IR014_reduced['XC'])[0:102400]).reshape(50, 2048))
data_C_IR014_reduced_DE = np.transpose(((data_C_IR014_reduced['XC'])[0:102400]).reshape(50, 2048))
data_A_IR014_original_DE = np.transpose(((data_A_IR014_original['X169_DE_time'])[0:102400]).reshape(50, 2048))
data_B_IR014_original_DE = np.transpose(((data_B_IR014_original['X170_DE_time'])[0:102400]).reshape(50, 2048))
data_C_IR014_original_DE = np.transpose(((data_C_IR014_original['X171_DE_time'])[0:102400]).reshape(50, 2048))
data_D_IR014_reduced_DE = np.hstack([np.hstack([data_A_IR014_reduced_DE, data_B_IR014_reduced_DE]), data_C_IR014_reduced_DE])
data_D_IR014_original_DE = np.hstack([np.hstack([data_A_IR014_original_DE, data_B_IR014_original_DE]), data_C_IR014_original_DE])


data_A_B014_reduced = loadmat("../Data/My Data/B014_0")
data_B_B014_reduced = loadmat("../Data/My Data/B014_1")
data_C_B014_reduced = loadmat("../Data/My Data/B014_2")
data_A_B014_original = loadmat("../Data/12k Drive End Bearing Fault Data/B014_0")
data_B_B014_original = loadmat("../Data/12k Drive End Bearing Fault Data/B014_1")
data_C_B014_original = loadmat("../Data/12k Drive End Bearing Fault Data/B014_2")
data_A_B014_reduced_DE = np.transpose(((data_A_B014_reduced['XC'])[0:102400]).reshape(50, 2048))
data_B_B014_reduced_DE = np.transpose(((data_B_B014_reduced['XC'])[0:102400]).reshape(50, 2048))
data_C_B014_reduced_DE = np.transpose(((data_C_B014_reduced['XC'])[0:102400]).reshape(50, 2048))
data_A_B014_original_DE = np.transpose(((data_A_B014_original['X185_DE_time'])[0:102400]).reshape(50, 2048))
data_B_B014_original_DE = np.transpose(((data_B_B014_original['X186_DE_time'])[0:102400]).reshape(50, 2048))
data_C_B014_original_DE = np.transpose(((data_C_B014_original['X187_DE_time'])[0:102400]).reshape(50, 2048))
data_D_B014_reduced_DE = np.hstack([np.hstack([data_A_B014_reduced_DE, data_B_B014_reduced_DE]), data_C_B014_reduced_DE])
data_D_B014_original_DE = np.hstack([np.hstack([data_A_B014_original_DE, data_B_B014_original_DE]), data_C_B014_original_DE])


data_A_OR014_reduced = loadmat("../Data/My Data/OR014@6_0")
data_B_OR014_reduced = loadmat("../Data/My Data/OR014@6_1")
data_C_OR014_reduced = loadmat("../Data/My Data/OR014@6_2")
data_A_OR014_original = loadmat("../Data/12k Drive End Bearing Fault Data/OR014@6_0")
data_B_OR014_original = loadmat("../Data/12k Drive End Bearing Fault Data/OR014@6_1")
data_C_OR014_original = loadmat("../Data/12k Drive End Bearing Fault Data/OR014@6_2")
data_A_OR014_reduced_DE = np.transpose(((data_A_OR014_reduced['XC'])[0:102400]).reshape(50, 2048))
data_B_OR014_reduced_DE = np.transpose(((data_B_OR014_reduced['XC'])[0:102400]).reshape(50, 2048))
data_C_OR014_reduced_DE = np.transpose(((data_C_OR014_reduced['XC'])[0:102400]).reshape(50, 2048))
data_A_OR014_original_DE = np.transpose(((data_A_OR014_original['X197_DE_time'])[0:102400]).reshape(50, 2048))
data_B_OR014_original_DE = np.transpose(((data_B_OR014_original['X198_DE_time'])[0:102400]).reshape(50, 2048))
data_C_OR014_original_DE = np.transpose(((data_C_OR014_original['X199_DE_time'])[0:102400]).reshape(50, 2048))
data_D_OR014_reduced_DE = np.hstack([np.hstack([data_A_OR014_reduced_DE, data_B_OR014_reduced_DE]), data_C_OR014_reduced_DE])
data_D_OR014_original_DE = np.hstack([np.hstack([data_A_OR014_original_DE, data_B_OR014_original_DE]), data_C_OR014_original_DE])


data_A_IR021_reduced = loadmat("../Data/My Data/IR021_0")
data_B_IR021_reduced = loadmat("../Data/My Data/IR021_1")
data_C_IR021_reduced = loadmat("../Data/My Data/IR021_2")
data_A_IR021_original = loadmat("../Data/12k Drive End Bearing Fault Data/IR021_0")
data_B_IR021_original = loadmat("../Data/12k Drive End Bearing Fault Data/IR021_1")
data_C_IR021_original = loadmat("../Data/12k Drive End Bearing Fault Data/IR021_2")
data_A_IR021_reduced_DE = np.transpose(((data_A_IR021_reduced['XC'])[0:102400]).reshape(50, 2048))
data_B_IR021_reduced_DE = np.transpose(((data_B_IR021_reduced['XC'])[0:102400]).reshape(50, 2048))
data_C_IR021_reduced_DE = np.transpose(((data_C_IR021_reduced['XC'])[0:102400]).reshape(50, 2048))
data_A_IR021_original_DE = np.transpose(((data_A_IR021_original['X209_DE_time'])[0:102400]).reshape(50, 2048))
data_B_IR021_original_DE = np.transpose(((data_B_IR021_original['X210_DE_time'])[0:102400]).reshape(50, 2048))
data_C_IR021_original_DE = np.transpose(((data_C_IR021_original['X211_DE_time'])[0:102400]).reshape(50, 2048))
data_D_IR021_reduced_DE = np.hstack([np.hstack([data_A_IR021_reduced_DE, data_B_IR021_reduced_DE]), data_C_IR021_reduced_DE])
data_D_IR021_original_DE = np.hstack([np.hstack([data_A_IR021_original_DE, data_B_IR021_original_DE]), data_C_IR021_original_DE])


data_A_B021_reduced = loadmat("../Data/My Data/B021_0")
data_B_B021_reduced = loadmat("../Data/My Data/B021_1")
data_C_B021_reduced = loadmat("../Data/My Data/B021_2")
data_A_B021_original = loadmat("../Data/12k Drive End Bearing Fault Data/B021_0")
data_B_B021_original = loadmat("../Data/12k Drive End Bearing Fault Data/B021_1")
data_C_B021_original = loadmat("../Data/12k Drive End Bearing Fault Data/B021_2")
data_A_B021_reduced_DE = np.transpose(((data_A_B021_reduced['XC'])[0:102400]).reshape(50, 2048))
data_B_B021_reduced_DE = np.transpose(((data_B_B021_reduced['XC'])[0:102400]).reshape(50, 2048))
data_C_B021_reduced_DE = np.transpose(((data_C_B021_reduced['XC'])[0:102400]).reshape(50, 2048))
data_A_B021_original_DE = np.transpose(((data_A_B021_original['X222_DE_time'])[0:102400]).reshape(50, 2048))
data_B_B021_original_DE = np.transpose(((data_B_B021_original['X223_DE_time'])[0:102400]).reshape(50, 2048))
data_C_B021_original_DE = np.transpose(((data_C_B021_original['X224_DE_time'])[0:102400]).reshape(50, 2048))
data_D_B021_reduced_DE = np.hstack([np.hstack([data_A_B021_reduced_DE, data_B_B021_reduced_DE]), data_C_B021_reduced_DE])
data_D_B021_original_DE = np.hstack([np.hstack([data_A_B021_original_DE, data_B_B021_original_DE]), data_C_B021_original_DE])


data_A_OR021_reduced = loadmat("../Data/My Data/OR021@6_0")
data_B_OR021_reduced = loadmat("../Data/My Data/OR021@6_1")
data_C_OR021_reduced = loadmat("../Data/My Data/OR021@6_2")
data_A_OR021_original = loadmat("../Data/12k Drive End Bearing Fault Data/OR021@6_0")
data_B_OR021_original = loadmat("../Data/12k Drive End Bearing Fault Data/OR021@6_1")
data_C_OR021_original = loadmat("../Data/12k Drive End Bearing Fault Data/OR021@6_2")
data_A_OR021_reduced_DE = np.transpose(((data_A_OR021_reduced['XC'])[0:102400]).reshape(50, 2048))
data_B_OR021_reduced_DE = np.transpose(((data_B_OR021_reduced['XC'])[0:102400]).reshape(50, 2048))
data_C_OR021_reduced_DE = np.transpose(((data_C_OR021_reduced['XC'])[0:102400]).reshape(50, 2048))
data_A_OR021_original_DE = np.transpose(((data_A_OR021_original['X234_DE_time'])[0:102400]).reshape(50, 2048))
data_B_OR021_original_DE = np.transpose(((data_B_OR021_original['X235_DE_time'])[0:102400]).reshape(50, 2048))
data_C_OR021_original_DE = np.transpose(((data_C_OR021_original['X236_DE_time'])[0:102400]).reshape(50, 2048))
data_D_OR021_reduced_DE = np.hstack([np.hstack([data_A_OR021_reduced_DE, data_B_OR021_reduced_DE]), data_C_OR021_reduced_DE])
data_D_OR021_original_DE = np.hstack([np.hstack([data_A_OR021_original_DE, data_B_OR021_original_DE]), data_C_OR021_original_DE])
print("Reading finished.")

min_max_scaler = MinMaxScaler()

data_A_reduced = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_A_normal_reduced_DE, data_A_IR007_reduced_DE]), data_A_B007_reduced_DE]), data_A_OR007_reduced_DE]), data_A_IR014_reduced_DE]), data_A_B014_reduced_DE]), data_A_OR014_reduced_DE]), data_A_IR021_reduced_DE]), data_A_B021_reduced_DE]), data_A_OR021_reduced_DE])
data_B_reduced = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_B_normal_reduced_DE, data_B_IR007_reduced_DE]), data_B_B007_reduced_DE]), data_B_OR007_reduced_DE]), data_B_IR014_reduced_DE]), data_B_B014_reduced_DE]), data_B_OR014_reduced_DE]), data_B_IR021_reduced_DE]), data_B_B021_reduced_DE]), data_B_OR021_reduced_DE])
data_C_reduced = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_C_normal_reduced_DE, data_C_IR007_reduced_DE]), data_C_B007_reduced_DE]), data_C_OR007_reduced_DE]), data_C_IR014_reduced_DE]), data_C_B014_reduced_DE]), data_C_OR014_reduced_DE]), data_C_IR021_reduced_DE]), data_C_B021_reduced_DE]), data_C_OR021_reduced_DE])
data_D_reduced = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_D_normal_reduced_DE, data_D_IR007_reduced_DE]), data_D_B007_reduced_DE]), data_D_OR007_reduced_DE]), data_D_IR014_reduced_DE]), data_D_B014_reduced_DE]), data_D_OR014_reduced_DE]), data_D_IR021_reduced_DE]), data_D_B021_reduced_DE]), data_D_OR021_reduced_DE])
data_A_original = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_A_normal_original_DE, data_A_IR007_original_DE]), data_A_B007_original_DE]), data_A_OR007_original_DE]), data_A_IR014_original_DE]), data_A_B014_original_DE]), data_A_OR014_original_DE]), data_A_IR021_original_DE]), data_A_B021_original_DE]), data_A_OR021_original_DE])
data_B_original = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_B_normal_original_DE, data_B_IR007_original_DE]), data_B_B007_original_DE]), data_B_OR007_original_DE]), data_B_IR014_original_DE]), data_B_B014_original_DE]), data_B_OR014_original_DE]), data_B_IR021_original_DE]), data_B_B021_original_DE]), data_B_OR021_original_DE])
data_C_original = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_C_normal_original_DE, data_C_IR007_original_DE]), data_C_B007_original_DE]), data_C_OR007_original_DE]), data_C_IR014_original_DE]), data_C_B014_original_DE]), data_C_OR014_original_DE]), data_C_IR021_original_DE]), data_C_B021_original_DE]), data_C_OR021_original_DE])
data_D_original = np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([np.hstack([data_D_normal_original_DE, data_D_IR007_original_DE]), data_D_B007_original_DE]), data_D_OR007_original_DE]), data_D_IR014_original_DE]), data_D_B014_original_DE]), data_D_OR014_original_DE]), data_D_IR021_original_DE]), data_D_B021_original_DE]), data_D_OR021_original_DE])

n_feature = 36
t_A = np.zeros((n_feature, 500))
t_B = np.zeros((n_feature, 500))
t_C = np.zeros((n_feature, 500))
t_D = np.zeros((n_feature, 1500))

temp_A_reduced = data_A_reduced+abs(np.min(data_A_reduced))+1
temp_B_reduced = data_B_reduced+abs(np.min(data_B_reduced))+1
temp_C_reduced = data_C_reduced+abs(np.min(data_C_reduced))+1
temp_D_reduced = data_D_reduced+abs(np.min(data_D_reduced))+1
temp_A_original = data_A_original+abs(np.min(data_A_original))+1
temp_B_original = data_B_original+abs(np.min(data_B_original))+1
temp_C_original = data_C_original+abs(np.min(data_C_original))+1
temp_D_original = data_D_original+abs(np.min(data_D_original))+1

for i in range(500):
    mean = sum(data_A_reduced[:, i])/2048
    t_A[0, i] = max(abs(data_A_reduced[:, i]))-min(abs(data_A_reduced[:, i]))
    t_A[1, i] = max(abs(data_A_reduced[:, i]))
    t_A[2, i] = mean
    t_A[3, i] = sum(pow(data_A_reduced[:, i], 2))/2048
    t_A[4, i] = pow(sum(pow(data_A_reduced[:, i], 2))/2048, 0.5)
    t_A[5, i] = sum(abs(data_A_reduced[:, i]))/2048
    t_A[6, i] = pow(sum(pow(abs(data_A_reduced[:, i]), 0.5))/2048, 2)
    t_A[7, i] = sum(pow(data_A_reduced[:, i]-mean, 2))/2048
    t_A[8, i] = pow(sum(pow(data_A_reduced[:, i]-mean, 2))/2048, 0.5)
    t_A[9, i] = sum(pow(data_A_reduced[:, i], 3))/2048
    t_A[10, i] = sum(pow(data_A_reduced[:, i], 4))/2048
    t_A[11, i] = sum(pow(data_A_reduced[:, i], 3))/2048/pow(pow(sum(pow(data_A_reduced[:, i], 2))/2048, 0.5), 3)
    t_A[12, i] = sum(pow(data_A_reduced[:, i], 4))/2048/pow(pow(sum(pow(data_A_reduced[:, i], 2))/2048, 0.5), 4)
    t_A[13, i] = max(abs(data_A_reduced[:, i]))/pow(sum(pow(data_A_reduced[:, i], 2))/2048, 0.5)
    t_A[14, i] = max(abs(data_A_reduced[:, i]))/(sum(abs(data_A_reduced[:, i]))/2048)
    t_A[15, i] = max(abs(data_A_reduced[:, i]))/pow(sum(pow(abs(data_A_reduced[:, i]), 0.5))/2048, 2)
    t_A[16, i] = pow(sum(pow(data_A_reduced[:, i], 2))/2048, 0.5)/(sum(abs(data_A_reduced[:, i]))/2048)
    t_A[17, i] = np.mean(np.multiply(temp_A_reduced[:, i], np.log(temp_A_reduced[:, i])))
    mean = sum(data_A_original[:, i]) / 2048
    t_A[18, i] = max(abs(data_A_original[:, i])) - min(abs(data_A_original[:, i]))
    t_A[19, i] = max(abs(data_A_original[:, i]))
    t_A[20, i] = mean
    t_A[21, i] = sum(pow(data_A_original[:, i], 2)) / 2048
    t_A[22, i] = pow(sum(pow(data_A_original[:, i], 2)) / 2048, 0.5)
    t_A[23, i] = sum(abs(data_A_original[:, i])) / 2048
    t_A[24, i] = pow(sum(pow(abs(data_A_original[:, i]), 0.5)) / 2048, 2)
    t_A[25, i] = sum(pow(data_A_original[:, i] - mean, 2)) / 2048
    t_A[26, i] = pow(sum(pow(data_A_original[:, i] - mean, 2)) / 2048, 0.5)
    t_A[27, i] = sum(pow(data_A_original[:, i], 3)) / 2048
    t_A[28, i] = sum(pow(data_A_original[:, i], 4)) / 2048
    t_A[29, i] = sum(pow(data_A_original[:, i], 3)) / 2048 / pow(pow(sum(pow(data_A_original[:, i], 2)) / 2048, 0.5), 3)
    t_A[30, i] = sum(pow(data_A_original[:, i], 4)) / 2048 / pow(pow(sum(pow(data_A_original[:, i], 2)) / 2048, 0.5), 4)
    t_A[31, i] = max(abs(data_A_original[:, i])) / pow(sum(pow(data_A_original[:, i], 2)) / 2048, 0.5)
    t_A[32, i] = max(abs(data_A_original[:, i])) / (sum(abs(data_A_original[:, i])) / 2048)
    t_A[33, i] = max(abs(data_A_original[:, i])) / pow(sum(pow(abs(data_A_original[:, i]), 0.5)) / 2048, 2)
    t_A[34, i] = pow(sum(pow(data_A_original[:, i], 2)) / 2048, 0.5) / (sum(abs(data_A_original[:, i])) / 2048)
    t_A[35, i] = np.mean(np.multiply(temp_A_original[:, i], np.log(temp_A_original[:, i])))

    mean = sum(data_B_reduced[:, i]) / 2048
    t_B[0, i] = max(abs(data_B_reduced[:, i])) - min(abs(data_B_reduced[:, i]))
    t_B[1, i] = max(abs(data_B_reduced[:, i]))
    t_B[2, i] = mean
    t_B[3, i] = sum(pow(data_B_reduced[:, i], 2)) / 2048
    t_B[4, i] = pow(sum(pow(data_B_reduced[:, i], 2)) / 2048, 0.5)
    t_B[5, i] = sum(abs(data_B_reduced[:, i])) / 2048
    t_B[6, i] = pow(sum(pow(abs(data_B_reduced[:, i]), 0.5)) / 2048, 2)
    t_B[7, i] = sum(pow(data_B_reduced[:, i] - mean, 2)) / 2048
    t_B[8, i] = pow(sum(pow(data_B_reduced[:, i] - mean, 2)) / 2048, 0.5)
    t_B[9, i] = sum(pow(data_B_reduced[:, i], 3)) / 2048
    t_B[10, i] = sum(pow(data_B_reduced[:, i], 4)) / 2048
    t_B[11, i] = sum(pow(data_B_reduced[:, i], 3)) / 2048 / pow(pow(sum(pow(data_B_reduced[:, i], 2)) / 2048, 0.5), 3)
    t_B[12, i] = sum(pow(data_B_reduced[:, i], 4)) / 2048 / pow(pow(sum(pow(data_B_reduced[:, i], 2)) / 2048, 0.5), 4)
    t_B[13, i] = max(abs(data_B_reduced[:, i])) / pow(sum(pow(data_B_reduced[:, i], 2)) / 2048, 0.5)
    t_B[14, i] = max(abs(data_B_reduced[:, i])) / (sum(abs(data_B_reduced[:, i])) / 2048)
    t_B[15, i] = max(abs(data_B_reduced[:, i])) / pow(sum(pow(abs(data_B_reduced[:, i]), 0.5)) / 2048, 2)
    t_B[16, i] = pow(sum(pow(data_B_reduced[:, i], 2)) / 2048, 0.5) / (sum(abs(data_B_reduced[:, i])) / 2048)
    t_B[17, i] = np.mean(np.multiply(temp_B_reduced[:, i], np.log(temp_B_reduced[:, i])))
    mean = sum(data_B_original[:, i]) / 2048
    t_B[18, i] = max(abs(data_B_original[:, i])) - min(abs(data_B_original[:, i]))
    t_B[19, i] = max(abs(data_B_original[:, i]))
    t_B[20, i] = mean
    t_B[21, i] = sum(pow(data_B_original[:, i], 2)) / 2048
    t_B[22, i] = pow(sum(pow(data_B_original[:, i], 2)) / 2048, 0.5)
    t_B[23, i] = sum(abs(data_B_original[:, i])) / 2048
    t_B[24, i] = pow(sum(pow(abs(data_B_original[:, i]), 0.5)) / 2048, 2)
    t_B[25, i] = sum(pow(data_B_original[:, i] - mean, 2)) / 2048
    t_B[26, i] = pow(sum(pow(data_B_original[:, i] - mean, 2)) / 2048, 0.5)
    t_B[27, i] = sum(pow(data_B_original[:, i], 3)) / 2048
    t_B[28, i] = sum(pow(data_B_original[:, i], 4)) / 2048
    t_B[29, i] = sum(pow(data_B_original[:, i], 3)) / 2048 / pow(pow(sum(pow(data_B_original[:, i], 2)) / 2048, 0.5), 3)
    t_B[30, i] = sum(pow(data_B_original[:, i], 4)) / 2048 / pow(pow(sum(pow(data_B_original[:, i], 2)) / 2048, 0.5), 4)
    t_B[31, i] = max(abs(data_B_original[:, i])) / pow(sum(pow(data_B_original[:, i], 2)) / 2048, 0.5)
    t_B[32, i] = max(abs(data_B_original[:, i])) / (sum(abs(data_B_original[:, i])) / 2048)
    t_B[33, i] = max(abs(data_B_original[:, i])) / pow(sum(pow(abs(data_B_original[:, i]), 0.5)) / 2048, 2)
    t_B[34, i] = pow(sum(pow(data_B_original[:, i], 2)) / 2048, 0.5) / (sum(abs(data_B_original[:, i])) / 2048)
    t_B[35, i] = np.mean(np.multiply(temp_B_original[:, i], np.log(temp_B_original[:, i])))

    mean = sum(data_C_reduced[:, i]) / 2048
    t_C[0, i] = max(abs(data_C_reduced[:, i])) - min(abs(data_C_reduced[:, i]))
    t_C[1, i] = max(abs(data_C_reduced[:, i]))
    t_C[2, i] = mean
    t_C[3, i] = sum(pow(data_C_reduced[:, i], 2)) / 2048
    t_C[4, i] = pow(sum(pow(data_C_reduced[:, i], 2)) / 2048, 0.5)
    t_C[5, i] = sum(abs(data_C_reduced[:, i])) / 2048
    t_C[6, i] = pow(sum(pow(abs(data_C_reduced[:, i]), 0.5)) / 2048, 2)
    t_C[7, i] = sum(pow(data_C_reduced[:, i] - mean, 2)) / 2048
    t_C[8, i] = pow(sum(pow(data_C_reduced[:, i] - mean, 2)) / 2048, 0.5)
    t_C[9, i] = sum(pow(data_C_reduced[:, i], 3)) / 2048
    t_C[10, i] = sum(pow(data_C_reduced[:, i], 4)) / 2048
    t_C[11, i] = sum(pow(data_C_reduced[:, i], 3)) / 2048 / pow(pow(sum(pow(data_C_reduced[:, i], 2)) / 2048, 0.5), 3)
    t_C[12, i] = sum(pow(data_C_reduced[:, i], 4)) / 2048 / pow(pow(sum(pow(data_C_reduced[:, i], 2)) / 2048, 0.5), 4)
    t_C[13, i] = max(abs(data_C_reduced[:, i])) / pow(sum(pow(data_C_reduced[:, i], 2)) / 2048, 0.5)
    t_C[14, i] = max(abs(data_C_reduced[:, i])) / (sum(abs(data_C_reduced[:, i])) / 2048)
    t_C[15, i] = max(abs(data_C_reduced[:, i])) / pow(sum(pow(abs(data_C_reduced[:, i]), 0.5)) / 2048, 2)
    t_C[16, i] = pow(sum(pow(data_C_reduced[:, i], 2)) / 2048, 0.5) / (sum(abs(data_C_reduced[:, i])) / 2048)
    t_C[17, i] = np.mean(np.multiply(temp_C_reduced[:, i], np.log(temp_C_reduced[:, i])))
    mean = sum(data_C_original[:, i]) / 2048
    t_C[18, i] = max(abs(data_C_original[:, i])) - min(abs(data_C_original[:, i]))
    t_C[19, i] = max(abs(data_C_original[:, i]))
    t_C[20, i] = mean
    t_C[21, i] = sum(pow(data_C_original[:, i], 2)) / 2048
    t_C[22, i] = pow(sum(pow(data_C_original[:, i], 2)) / 2048, 0.5)
    t_C[23, i] = sum(abs(data_C_original[:, i])) / 2048
    t_C[24, i] = pow(sum(pow(abs(data_C_original[:, i]), 0.5)) / 2048, 2)
    t_C[25, i] = sum(pow(data_C_original[:, i] - mean, 2)) / 2048
    t_C[26, i] = pow(sum(pow(data_C_original[:, i] - mean, 2)) / 2048, 0.5)
    t_C[27, i] = sum(pow(data_C_original[:, i], 3)) / 2048
    t_C[28, i] = sum(pow(data_C_original[:, i], 4)) / 2048
    t_C[29, i] = sum(pow(data_C_original[:, i], 3)) / 2048 / pow(pow(sum(pow(data_C_original[:, i], 2)) / 2048, 0.5), 3)
    t_C[30, i] = sum(pow(data_C_original[:, i], 4)) / 2048 / pow(pow(sum(pow(data_C_original[:, i], 2)) / 2048, 0.5), 4)
    t_C[31, i] = max(abs(data_C_original[:, i])) / pow(sum(pow(data_C_original[:, i], 2)) / 2048, 0.5)
    t_C[32, i] = max(abs(data_C_original[:, i])) / (sum(abs(data_C_original[:, i])) / 2048)
    t_C[33, i] = max(abs(data_C_original[:, i])) / pow(sum(pow(abs(data_C_original[:, i]), 0.5)) / 2048, 2)
    t_C[34, i] = pow(sum(pow(data_C_original[:, i], 2)) / 2048, 0.5) / (sum(abs(data_C_original[:, i])) / 2048)
    t_C[35, i] = np.mean(np.multiply(temp_C_original[:, i], np.log(temp_C_original[:, i])))

for i in range(1500):
    mean = sum(data_D_reduced[:, i]) / 2048
    t_D[0, i] = max(abs(data_D_reduced[:, i])) - min(abs(data_D_reduced[:, i]))
    t_D[1, i] = max(abs(data_D_reduced[:, i]))
    t_D[2, i] = mean
    t_D[3, i] = sum(pow(data_D_reduced[:, i], 2)) / 2048
    t_D[4, i] = pow(sum(pow(data_D_reduced[:, i], 2)) / 2048, 0.5)
    t_D[5, i] = sum(abs(data_D_reduced[:, i])) / 2048
    t_D[6, i] = pow(sum(pow(abs(data_D_reduced[:, i]), 0.5)) / 2048, 2)
    t_D[7, i] = sum(pow(data_D_reduced[:, i] - mean, 2)) / 2048
    t_D[8, i] = pow(sum(pow(data_D_reduced[:, i] - mean, 2)) / 2048, 0.5)
    t_D[9, i] = sum(pow(data_D_reduced[:, i], 3)) / 2048
    t_D[10, i] = sum(pow(data_D_reduced[:, i], 4)) / 2048
    t_D[11, i] = sum(pow(data_D_reduced[:, i], 3)) / 2048 / pow(pow(sum(pow(data_D_reduced[:, i], 2)) / 2048, 0.5), 3)
    t_D[12, i] = sum(pow(data_D_reduced[:, i], 4)) / 2048 / pow(pow(sum(pow(data_D_reduced[:, i], 2)) / 2048, 0.5), 4)
    t_D[13, i] = max(abs(data_D_reduced[:, i])) / pow(sum(pow(data_D_reduced[:, i], 2)) / 2048, 0.5)
    t_D[14, i] = max(abs(data_D_reduced[:, i])) / (sum(abs(data_D_reduced[:, i])) / 2048)
    t_D[15, i] = max(abs(data_D_reduced[:, i])) / pow(sum(pow(abs(data_D_reduced[:, i]), 0.5)) / 2048, 2)
    t_D[16, i] = pow(sum(pow(data_D_reduced[:, i], 2)) / 2048, 0.5) / (sum(abs(data_D_reduced[:, i])) / 2048)
    t_D[17, i] = np.mean(np.multiply(temp_D_reduced[:, i], np.log(temp_D_reduced[:, i])))
    mean = sum(data_D_original[:, i]) / 2048
    t_D[18, i] = max(abs(data_D_original[:, i])) - min(abs(data_D_original[:, i]))
    t_D[19, i] = max(abs(data_D_original[:, i]))
    t_D[20, i] = mean
    t_D[21, i] = sum(pow(data_D_original[:, i], 2)) / 2048
    t_D[22, i] = pow(sum(pow(data_D_original[:, i], 2)) / 2048, 0.5)
    t_D[23, i] = sum(abs(data_D_original[:, i])) / 2048
    t_D[24, i] = pow(sum(pow(abs(data_D_original[:, i]), 0.5)) / 2048, 2)
    t_D[25, i] = sum(pow(data_D_original[:, i] - mean, 2)) / 2048
    t_D[26, i] = pow(sum(pow(data_D_original[:, i] - mean, 2)) / 2048, 0.5)
    t_D[27, i] = sum(pow(data_D_original[:, i], 3)) / 2048
    t_D[28, i] = sum(pow(data_D_original[:, i], 4)) / 2048
    t_D[29, i] = sum(pow(data_D_original[:, i], 3)) / 2048 / pow(pow(sum(pow(data_D_original[:, i], 2)) / 2048, 0.5), 3)
    t_D[30, i] = sum(pow(data_D_original[:, i], 4)) / 2048 / pow(pow(sum(pow(data_D_original[:, i], 2)) / 2048, 0.5), 4)
    t_D[31, i] = max(abs(data_D_original[:, i])) / pow(sum(pow(data_D_original[:, i], 2)) / 2048, 0.5)
    t_D[32, i] = max(abs(data_D_original[:, i])) / (sum(abs(data_D_original[:, i])) / 2048)
    t_D[33, i] = max(abs(data_D_original[:, i])) / pow(sum(pow(abs(data_D_original[:, i]), 0.5)) / 2048, 2)
    t_D[34, i] = pow(sum(pow(data_D_original[:, i], 2)) / 2048, 0.5) / (sum(abs(data_D_original[:, i])) / 2048)
    t_D[35, i] = np.mean(np.multiply(temp_D_original[:, i], np.log(temp_D_original[:, i])))
print("Extraction finished.")

data_A_minmax = min_max_scaler.fit_transform(np.transpose(t_A))
data_B_minmax = min_max_scaler.fit_transform(np.transpose(t_B))
data_C_minmax = min_max_scaler.fit_transform(np.transpose(t_C))
data_D_minmax = min_max_scaler.fit_transform(np.transpose(t_D))

import warnings
warnings.filterwarnings("ignore")
plt.figure(1)
for k in range(n_feature):
    plt.subplot(n_feature/2, 2, k+1)
    plt.xticks([])
    plt.yticks([])
    for i in range(10):
        sns.distplot(t_D[k, (150*i):(150*(i+1))], rug=True, hist=False)
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
tra = np.zeros((epoches, 1000))
tea = np.zeros((epoches, 1000))
erra = np.zeros((epoches, 300))
sum_tr = 0
sum_te = 0
for epoch in range(epoches):
    print("Current epoch ", epoch)
    X_train, X_test, y_train, y_test = train_test_split(data_D_minmax, eval_D, test_size=0.4)
    RBM_hidden_sizes = [18, 13, 10]
    inpX = X_train
    rbm_list = []
    input_size = inpX.shape[1]
    '''
    for i, size in enumerate(RBM_hidden_sizes):
        print('SSRBM: ', i, ' ', input_size, '->', size)
        rbm_list.append(SSRBM(input_size, size, 100, 0.3, 30, 1))
        input_size = size

    S = np.zeros([X_train.shape[0], X_train.shape[0]])
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[0]):
            if y_train[i].all() == y_train[j].all():
                S[i, j] = 0
            else:
                S[i, j] = np.matmul(X_train[i], np.transpose(X_train[j])) / (np.power(np.sum(np.power(X_train[i], 2)), 0.5) * np.power(np.sum(np.power(X_train[j], 2)), 0.5))
    count = 0
    for rbm in rbm_list:
        print('New SSRBM:')
        err = rbm.train(inpX, S)
        for k in range(len(err)):
            erra[epoch, 100*count+k] = err[k]
        U, sigma, VT = np.linalg.svd(S)
        Sigma = np.zeros([inpX.shape[1], inpX.shape[1]])
        for i in range(inpX.shape[1]):
            Sigma[i, i] = sigma[i]
        X_ = np.transpose(np.matmul(np.sqrt(Sigma), np.transpose(U[0:inpX.shape[0], 0:inpX.shape[1]])))
        inpX = rbm.rbm_outpt(inpX, X_)
        count += 1

    nNet = NN(RBM_hidden_sizes, X_train, y_train, 1, 1000, 30)
    nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
    tr, te, result_tr, result_te = nNet.train(X_test, y_test)
    tra[epoch] = result_tr
    tea[epoch] = result_te

    '''

    #'''
    for i, size in enumerate(RBM_hidden_sizes):
        print('RBM: ', i, ' ', input_size, '->', size)
        rbm_list.append(RBM(input_size, size, 100, 0.3, 30))
        input_size = size
    
    count = 0
    for rbm in rbm_list:
        print('New RBM:')
        err = rbm.train(inpX)
        for k in range(len(err)):
            erra[epoch, 100*count+k] = err[k]
        inpX = rbm.rbm_outpt(inpX)
        count += 1
    
    nNet = NN(RBM_hidden_sizes, X_train, y_train, 1, 1000, 30)
    nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
    tr, te, result_tr, result_te = nNet.train(X_test, y_test)
    tra[epoch] = result_tr
    tea[epoch] = result_te
    
    #'''
'''
ave_erra = []
for i in range(300):
    Sum_erra = 0
    for j in range(epoches):
        Sum_erra = Sum_erra + erra[j][i]
    ave_erra.append(Sum_erra / epoches)
np.save('./result/error_SSDBN.npy', ave_erra)
plt.plot(range(300), ave_erra)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
'''
#'''
ave_tra = []
ave_tea = []
for i in range(1000):
    Sum_tra = 0
    Sum_tea = 0
    for j in range(epoches):
        Sum_tra = Sum_tra + tra[j][i]
        Sum_tea = Sum_tea + tea[j][i]
    ave_tra.append(Sum_tra / epoches)
    ave_tea.append(Sum_tea / epoches)
np.save('./result/tr_DBN_isigmoid_6_0.2.npy', ave_tra)
np.save('./result/te_DBN_isigmoid_6_0.2.npy', ave_tea)
label = ['Training Dataset', 'Testing Dataset']
plt.plot(range(1000), ave_tra)
plt.plot(range(1000), ave_tea)
plt.legend(label)
plt.xlabel('Epoch')
plt.ylabel('Accuracy Rate')
plt.show()
#'''