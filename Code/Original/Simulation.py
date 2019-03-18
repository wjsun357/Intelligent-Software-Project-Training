#定义故障诊断问题和故障类型
#对故障信号进行［０，１ ］标准化处理
#划分数据集为训练集和测试集
#初始化ＤＢＮ 的相关参数
#用训练集训练堆叠ＲＢＭ
#测试集输入到已训练的堆叠ＲＢＭ模型中，并记录每个隐含层的输出向量
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
'''
t=[]
x1=[]
x2=[]
x=[]
z1=[]
z2=[]
z=[]
y1=[]
y2=[]
y3=[]
y4=[]
y=[]
for i in range(256000):
    time=i/12800.0
    t.append(time)
    x1_append=5*math.cos(20*math.pi*time)+10*math.cos(40*math.pi*time)
    x1.append(x1_append)
    x2_append=15*math.cos(60*math.pi*time)+20*math.cos(80*math.pi*time)
    x2.append(x2_append)
    w=20*np.random.normal()
    x.append(x1_append+x2_append+w) 
    z1_append=4*math.sin(25*math.pi*time)*math.sin(30*math.pi*time)+math.sin(40*math.pow(math.pi,2)*time)
    z1.append(z1_append)
    z2_append=(10+5*math.cos(10*math.pi*time))*math.cos(2*math.pi*time+2*math.cos(5*math.pi*time))
    z2.append(z2_append)
    w=20*np.random.normal()
    z.append(z1_append+z2_append+w)
    y1_append=5*(1+math.cos(4*math.pi*time))*math.cos(20*math.pi*time)
    y1.append(y1_append)
    y2_append=10*(1+math.cos(4*math.pi*time))*math.cos(40*math.pi*time)
    y2.append(y2_append)
    y3_append=15*(1+math.cos(4*math.pi*time))*math.cos(60*math.pi*time)
    y3.append(y3_append)
    y4_append=20*(1+math.cos(4*math.pi*time))*math.cos(80*math.pi*time)
    y4.append(y4_append)
    w=20*np.random.normal()
    y.append(y1_append+y2_append+y3_append+y4_append+w)

samplex=np.zeros((512,500))
sampley=np.zeros((512,500))
samplez=np.zeros((512,500))

for i in range(0,512):
    for j in range(0,500):
        samplex[i][j]=x[500*i+j]
        sampley[i][j]=x[500*i+j]
        samplez[i][j]=x[500*i+j]

min_max_scaler = MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(samplex)
y_minmax = min_max_scaler.fit_transform(sampley)
z_minmax = min_max_scaler.fit_transform(samplez)

x_train, x_test = train_test_split(x_minmax, test_size=0.4)
y_train, y_test = train_test_split(y_minmax, test_size=0.4)
z_train, z_test = train_test_split(z_minmax, test_size=0.4)
'''