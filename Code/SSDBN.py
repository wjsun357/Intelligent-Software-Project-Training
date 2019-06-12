import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import isigmoid
from sklearn.model_selection import train_test_split
# Image library for image manipulation
# import Image
# Utils file
from tensorflow.examples.tutorials.mnist import input_data


# 定义SSRBM
class SSRBM(object):
    def __init__(self, input_size, output_size, epoches, learning_rate, batchsize, proportion):
        # 定义超参数
        self._input_size = input_size  # 输入大小
        self._output_size = output_size  # 输出大小
        self._epoches = epoches  # 迭代次数
        self._learning_rate = learning_rate  # 学习率
        self._batchsize = batchsize  # 抽样数
        self._proportion = proportion  # 比例

        # 初始权重和偏差
        self.w = 0.1 * np.random.randn(input_size, output_size)  # 权重
        self.hb = np.zeros([output_size], np.float64)  # 隐藏层偏差，输出
        self.vb = np.zeros([input_size], np.float64)  # 可视层偏差，输入
        self.ub = np.zeros([input_size], np.float64)
        self.p = 0.1 * np.random.randn(input_size, output_size)

    # 隐含层条件概率
    def prob_h_given_vu(self, visible, u, p, w, hb):
        # Sigmoid
        return tf.nn.sigmoid(tf.matmul(visible, w) + self._proportion * tf.matmul(u, p) + hb)
        # return isigmoid.my_sigmoid_tf(tf.matmul(visible, w) + hb)  # matmul，矩阵乘法

    # 可视层条件概率
    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)
        # return isigmoid.my_sigmoid_tf(tf.matmul(hidden, tf.transpose(w)) + vb)

    # 样本概率
    def sample_prob(self, probs):
        # 要么是0，要么是1
        return tf.nn.relu(tf.sign(probs - tf.cast(tf.random_uniform(tf.shape(probs)), np.float64)))

    def prob_u_given_h(self, hidden, p, ub):
        return tf.nn.sigmoid(self._proportion * (tf.matmul(hidden, tf.transpose(p)) + ub))

    def train(self, X, S):  # X为总特征数
        # 创建placeholder
        _w = tf.placeholder(tf.float64, [self._input_size, self._output_size])
        _hb = tf.placeholder(tf.float64, [self._output_size])
        _vb = tf.placeholder(tf.float64, [self._input_size])
        _ub = tf.placeholder(tf.float64, [self._input_size])
        _p = tf.placeholder(tf.float64, [self._input_size, self._output_size])

        prv_w = self.w
        prv_hb = self.hb
        prv_vb = self.vb
        prv_ub = self.ub
        prv_p = self.p

        cur_w = self.w
        cur_hb = self.hb
        cur_vb = self.vb
        cur_ub = self.ub
        cur_p = self.p

        v0 = tf.placeholder(tf.float64, [None, self._input_size])
        u0 = tf.placeholder(tf.float64, [None, self._input_size])

        # 初始样本概率
        temp_h0 = self.prob_h_given_vu(v0, u0, _p, _w, _hb)
        h0 = self.sample_prob(temp_h0)  # 0或1
        temp_v1 = self.prob_v_given_h(h0, _w, _vb)
        v1 = self.sample_prob(temp_v1)  # 0或1
        temp_u1 = self.prob_u_given_h(h0, _p, _ub)
        u1 = self.sample_prob(temp_u1)
        h1 = self.prob_h_given_vu(v1, u1, _p, _w, _hb)

        positive_grad = tf.matmul(tf.transpose(v0), temp_h0)  # 取决于观测值，正阶段增加训练数据的可能性
        negative_grad = tf.matmul(tf.transpose(v1), h1)  # 只取决于模型，负阶段减少由模型生成的样本的概率
        positive_grad_u = tf.matmul(tf.transpose(u0), temp_h0)
        negative_grad_u = tf.matmul(tf.transpose(u1), h1)

        # (positive_grad - negative_grad) / tf.cast(tf.shape(v0)[0], np.float64)为对比散度
        update_w = _w + self._learning_rate * (positive_grad - negative_grad) / tf.cast(tf.shape(v0)[0], np.float64)
        update_vb = _vb + self._learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + self._learning_rate * tf.reduce_mean(temp_h0 - h1, 0)
        update_ub = _ub + self._learning_rate * tf.reduce_mean(self._proportion * (u0 - u1), 0)
        update_p = _w + self._learning_rate * self._proportion * (positive_grad_u - negative_grad_u) / tf.cast(tf.shape(u0)[0], np.float64)
        # 错误值
        #err = tf.reduce_mean(tf.abs(v1 - u0))
        err = tf.reduce_sum(tf.square(v0 - v1))

        U, sigma, VT = np.linalg.svd(S)
        Sigma = np.zeros([X.shape[1], X.shape[1]])
        for i in range(X.shape[1]):
            Sigma[i, i] = sigma[i]
        batch2 = np.transpose(np.matmul(np.sqrt(Sigma), np.transpose(U[0:self._batchsize, 0:X.shape[1]])))

        # 循环
        error_array = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self._epoches):
                for start, end in zip(range(0, len(X), self._batchsize), range(self._batchsize, len(X), self._batchsize)):
                    # [0,2048,100] [100,2048,100]，zip后得到[(0,100),(100,200),(200,300),...]
                    batch1 = X[start:end]
                    # 则batch为X[0:100],[100:200],...
                    # 更新
                    cur_w = sess.run(update_w, feed_dict={v0: batch1, u0: batch2, _w: prv_w, _hb: prv_hb, _vb: prv_vb, _ub: prv_ub, _p: prv_p})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch1, u0: batch2, _w: prv_w, _hb: prv_hb, _vb: prv_vb, _ub: prv_ub, _p: prv_p})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch1, u0: batch2, _w: prv_w, _hb: prv_hb, _vb: prv_vb, _ub: prv_ub, _p: prv_p})
                    cur_ub = sess.run(update_ub, feed_dict={v0: batch1, u0: batch2, _w: prv_w, _hb: prv_hb, _vb: prv_vb, _ub: prv_ub, _p: prv_p})
                    cur_p = sess.run(update_p, feed_dict={v0: batch1, u0: batch2, _w: prv_w, _hb: prv_hb, _vb: prv_vb, _ub: prv_ub, _p: prv_p})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                    prv_ub = cur_ub
                    prv_p = cur_p
                X_ = np.transpose(np.matmul(np.sqrt(Sigma), np.transpose(U[0:X.shape[0], 0:X.shape[1]])))
                error = sess.run(err, feed_dict={v0: X, u0: X_, _w: cur_w, _vb: cur_vb, _hb: cur_hb, _ub: cur_ub, _p: cur_p})
                print('Epoch: %d' % epoch, 'reconstruction error: %f' % error)
                error_array.append(error)
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb
            self.ub = prv_ub
            self.p = prv_p
        return error_array

    def rbm_outpt(self, X1, X2):
        input_X1 = tf.constant(X1)
        input_X2 = tf.constant(X2)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        _p = tf.constant(self.p)
        out = tf.nn.sigmoid(tf.matmul(input_X1, _w) + self._proportion * tf.matmul(input_X2, _p) + _hb)
        # out = isigmoid.my_sigmoid_tf(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)


class NN(object):
    def __init__(self, sizes, X, Y, learning_rate, epoches, batchsize):
        # 超参数
        self._sizes = sizes
        self._X = X
        self._Y = Y
        self.w_list = []
        self.b_list = []
        self._learning_rate = learning_rate
        self._epoches = epoches
        self._batchsize = batchsize
        input_size = X.shape[1]  # 特征数

        # 循环初始化
        for size in self._sizes + [Y.shape[1]]:
            # Define upper limit for the uniform distribution range
            # max_range = 4 * math.sqrt(6. / (input_size + size))
            # 初始化权重，随机均匀分布
            self.w_list.append(0.1 * np.random.randn(input_size, size))
            # 初始化偏差
            self.b_list.append(np.zeros([size], np.float64))
            input_size = size

    def load_from_rbms(self, dbn_sizes, rbm_list):
        # 检查
        assert len(dbn_sizes) == len(self._sizes)

        for i in range(len(self._sizes)):
            assert dbn_sizes[i] == self._sizes[i]

        for i in range(len(self._sizes)):
            self.w_list[i] = rbm_list[i].w
            self.b_list[i] = rbm_list[i].hb

    def train(self, test_X, test_Y):
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _a[0] = tf.placeholder(tf.float64, [None, self._X.shape[1]])
        y = tf.placeholder(tf.float64, [None, self._Y.shape[1]])

        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            #_a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])
            _a[i] = isigmoid.my_sigmoid_tf(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])

        # _a[-1] = tf.nn.softmax(_a[-1])
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=_a[-1], labels=y))
        cost = tf.reduce_mean(tf.square(_a[-1] - y))

        _momentum = tf.placeholder(tf.float64, shape=[])

        train_op = tf.train.MomentumOptimizer(self._learning_rate, momentum=_momentum).minimize(cost)

        # Prediction operation
        # predict_op = tf.argmax(tf.nn.softmax(_a[-1]), 1)
        predict_op = tf.argmax(_a[-1], 1)  # 求得_a的最后一行的最大值的索引
        # 循环
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tr = []
            te = []
            for i in range(self._epoches):
                for start, end in zip(range(0, len(self._X), self._batchsize), range(self._batchsize, len(self._X), self._batchsize)):
                    # Run the training operation on the input data
                    if i>(int)(self._epoches/3*2):
                        sess.run(train_op, feed_dict={_a[0]: self._X[start:end], y: self._Y[start:end], _momentum: 0.5})
                    elif i>(int)(self._epoches/3):
                        sess.run(train_op, feed_dict={_a[0]: self._X[start:end], y: self._Y[start:end], _momentum: 0.9})
                    else:
                        sess.run(train_op, feed_dict={_a[0]: self._X[start:end], y: self._Y[start:end], _momentum: 0.9})
                for j in range(len(self._sizes) + 1):
                    # Retrieve weights and biases
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])
                print("Accuracy rating for epoch " + str(i) + ": " + str(np.mean(np.argmax(self._Y, axis=1) == sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y}))))
                print("Accuracy rating for testing dataset: " + str(np.mean(np.argmax(test_Y, axis=1) == sess.run(predict_op, feed_dict={_a[0]: test_X, y: test_Y}))))
                tr.append(np.mean(np.argmax(self._Y, axis=1) == sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y})))
                te.append(np.mean(np.argmax(test_Y, axis=1) == sess.run(predict_op, feed_dict={_a[0]: test_X, y: test_Y})))
                # print(sess.run(_a[-1], feed_dict={_a[0]: self._X, y: self._Y}))
            '''
            label = ['Training Dataset', 'Testing Dataset']
            plt.plot(range(self._epoches), tr)
            plt.plot(range(self._epoches), te)
            plt.legend(label)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy Rate')
            plt.show()
            '''
        return max(tr), max(te), tr, te


if __name__ == '__main__':
    # Loading in the mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels
    trX = trX.astype(np.float64)
    trY = trY.astype(np.float64)
    teX = teX.astype(np.float64)
    teY = teY.astype(np.float64)
    RBM_hidden_sizes = [200, 200, 10]  # create 4 layers of SSRBM with size 785-500-200-50
    # Since we are training, set input as training data
    trX, X_A_test, trY, y_A_test = train_test_split(trX, trY, test_size=0.98, random_state=0)
    teX, X_A_test, teY, y_A_test = train_test_split(teX, teY, test_size=0.9, random_state=0)
    inpX = trX
    # Create list to hold our RBMs
    rbm_list = []
    # Size of inputs is the number of inputs in the training set
    input_size = inpX.shape[1]

    # For each SSRBM we want to generate
    for i, size in enumerate(RBM_hidden_sizes):
        print('SSRBM: ', i, ' ', input_size, '->', size)
        rbm_list.append(SSRBM(input_size, size, 50, 1.0, 10, 0.5))
        input_size = size

    S = np.zeros([inpX.shape[0], inpX.shape[0]])
    for i in range(inpX.shape[0]):
        for j in range(inpX.shape[0]):
            if trY[i].all() == trY[j].all():
                S[i, j] = 0
            else:
                S[i, j] = np.matmul(inpX[i], np.transpose(inpX[j])) / (np.power(np.sum(np.power(inpX[i], 2)), 0.5) * np.power(np.sum(np.power(inpX[j], 2)), 0.5))

    # For each RBM in our list
    for rbm in rbm_list:
        print('New RBM:')
        # Train a new one
        rbm.train(inpX, S)
        U, sigma, VT = np.linalg.svd(S)
        Sigma = np.zeros([inpX.shape[1], inpX.shape[1]])
        for i in range(inpX.shape[1]):
            Sigma[i, i] = sigma[i]
        X_ = np.transpose(np.matmul(np.sqrt(Sigma), np.transpose(U[0:inpX.shape[0], 0:inpX.shape[1]])))
        # Return the output layer
        inpX = rbm.rbm_outpt(inpX, X_)

    nNet = NN(RBM_hidden_sizes, trX, trY, 1.0, 0.9, 200, 10)
    nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
    nNet.train(teX, teY)
