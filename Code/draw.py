import matplotlib.pyplot as plt
import numpy as np
plt.tick_params(labelsize=23)
tr_normal_sigmoid = np.load('./result/tr_normal_sigmoid.npy')
te_normal_sigmoid = np.load('./result/te_normal_sigmoid.npy')
tr_normal_isigmoid = np.load('./result/tr_normal_isigmoid.npy')
te_normal_isigmoid = np.load('./result/te_normal_isigmoid.npy')
tr_DBN_sigmoid = np.load('./result/tr_DBN_sigmoid.npy')
te_DBN_sigmoid = np.load('./result/te_DBN_sigmoid.npy')
tr_DBN_isigmoid = np.load('./result/tr_DBN_isigmoid.npy')
te_DBN_isigmoid = np.load('./result/te_DBN_isigmoid.npy')
tr_SSDBN_sigmoid = np.load('./result/tr_SSDBN_sigmoid.npy')
te_SSDBN_sigmoid = np.load('./result/te_SSDBN_sigmoid.npy')
tr_SSDBN_isigmoid = np.load('./result/tr_SSDBN_isigmoid.npy')
te_SSDBN_isigmoid = np.load('./result/te_SSDBN_isigmoid.npy')
'''
label = ['DBN (Training, Sigmoid)', 'DBN (Testing, Sigmoid)', 'SSDBN (Trainging, Sigmoid)', 'SSDBN (Testing, Sigmoid)']
plt.plot(range(0, 1000, 10), tr_DBN_sigmoid[0:1000:10], color='r')
plt.plot(range(0, 1000, 10), te_DBN_sigmoid[0:1000:10], color='r', marker='x')
plt.plot(range(0, 1000, 10), tr_SSDBN_sigmoid[0:1000:10], color='k')
plt.plot(range(0, 1000, 10), te_SSDBN_sigmoid[0:1000:10], color='k', marker='x')
plt.legend(label)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=30)
plt.xlabel('Epoch', fontsize=30)
plt.ylabel('Accuracy Rate', fontsize=30)
plt.show()
'''