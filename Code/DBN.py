import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Image library for image manipulation
# import Image
# Utils file
from tensorflow.examples.tutorials.mnist import input_data
# 定义RBM
class RBM(object):
    def __init__(self, input_size, output_size, epoches, learning_rate, batchsize):
        # 定义超参数
        self._input_size = input_size  # 输入大小
        self._output_size = output_size  # 输出大小
        self._epoches = epoches  # 迭代次数
        self._learning_rate = learning_rate  # 学习率
        self._batchsize = batchsize  # 抽样数

        # 初始权重和偏差
        self.w = np.zeros([input_size, output_size], np.float64)  # 权重
        self.hb = np.zeros([output_size], np.float64)  # 隐藏层偏差，输出
        self.vb = np.zeros([input_size], np.float64)  # 可视层偏差，输入

    def prob_h_given_v(self, visible, w, hb):
        # Sigmoid
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)  # matmul，矩阵乘法

    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    # 样本概率
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.cast(tf.random_uniform(tf.shape(probs)), np.float64)))

    def train(self, X):  # X为总特征数
        # 创建placeholder
        _w = tf.placeholder(tf.float64, [self._input_size, self._output_size])
        _hb = tf.placeholder(tf.float64, [self._output_size])
        _vb = tf.placeholder(tf.float64, [self._input_size])

        prv_w = np.zeros([self._input_size, self._output_size], np.float64)
        prv_hb = np.zeros([self._output_size], np.float64)
        prv_vb = np.zeros([self._input_size], np.float64)

        cur_w = np.zeros([self._input_size, self._output_size], np.float64)
        cur_hb = np.zeros([self._output_size], np.float64)
        cur_vb = np.zeros([self._input_size], np.float64)
        v0 = tf.placeholder(tf.float64, [None, self._input_size])

        # 初始样本概率
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)

        positive_grad = tf.matmul(tf.transpose(v0), h0)  # 取决于观测值，正阶段增加训练数据的可能性
        negative_grad = tf.matmul(tf.transpose(v1), h1)  # 只取决于模型，负阶段减少由模型生成的样本的概率

        # (positive_grad - negative_grad) / tf.cast(tf.shape(v0)[0], np.float64)为对比散度
        update_w = _w + self._learning_rate * (positive_grad - negative_grad) / tf.cast(tf.shape(v0)[0], np.float64)
        update_vb = _vb + self._learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + self._learning_rate * tf.reduce_mean(h0 - h1, 0)

        # 错误率
        err = tf.reduce_mean(tf.square(v0 - v1))

        # 循环
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self._epoches):
                for start, end in zip(range(0, len(X), self._batchsize), range(self._batchsize, len(X), self._batchsize)):
                # [0,2048,100] [100,2048,100]
                # [0,100] [100,200] [200,300]...    
                    batch = X[start:end]
                    # 更新
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error = sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
                print('Epoch: %d' % epoch, 'reconstruction error: %f' % error)
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb

    def rbm_outpt(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)

class NN(object):
    def __init__(self, sizes, X, Y, learning_rate, momentum, epoches, batchsize):
        # 超参数
        self._sizes = sizes
        self._X = X
        self._Y = Y
        self.w_list = []
        self.b_list = []
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._epoches = epoches
        self._batchsize = batchsize
        input_size = X.shape[1]  # 特征数

        # 循环初始化
        for size in self._sizes + [Y.shape[1]]:
            # Define upper limit for the uniform distribution range
            max_range = 4 * math.sqrt(6. / (input_size + size))
            # 初始化权重，随机均匀分布
            self.w_list.append(np.random.uniform(-max_range, max_range, [input_size, size]).astype(np.float64))
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
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])

        cost = tf.reduce_mean(tf.square(_a[-1] - y))

        train_op = tf.train.MomentumOptimizer(self._learning_rate, self._momentum).minimize(cost)

        # Prediction operation
        predict_op = tf.argmax(_a[-1], 1)

        # 循环
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tr = []
            te = []
            for i in range(self._epoches):
                for start, end in zip(range(0, len(self._X), self._batchsize), range(self._batchsize, len(self._X), self._batchsize)):
                    # Run the training operation on the input data
                    sess.run(train_op, feed_dict={_a[0]: self._X[start:end], y: self._Y[start:end]})
                for j in range(len(self._sizes) + 1):
                    # Retrieve weights and biases
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])
                print("Accuracy rating for epoch " + str(i) + ": " + str(np.mean(np.argmax(self._Y, axis=1) == sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y}))))
                print("Accuracy rating for testing dataset: " + str(np.mean(np.argmax(test_Y, axis=1) == sess.run(predict_op, feed_dict={_a[0]: test_X, y: test_Y}))))
                tr.append(np.mean(np.argmax(self._Y, axis=1) == sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y})))
                te.append(np.mean(np.argmax(test_Y, axis=1) == sess.run(predict_op, feed_dict={_a[0]: test_X, y: test_Y})))
            label = ['Training Dataset', 'Testing Dataset']
            plt.plot(range(self._epoches),tr)
            plt.plot(range(self._epoches),te)
            plt.legend(label)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy Rate')
            plt.show()
    '''
    def predict(self, test_X, test_Y):
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _a[0] = tf.placeholder(tf.float64, [None, test_X.shape[1]])
        y = tf.placeholder(tf.float64, [None, test_Y.shape[1]])

        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])

        predict_op = tf.argmax(_a[-1], 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("Accuracy rating for testing dataset: " + str(np.mean(np.argmax(test_Y, axis=1) == sess.run(predict_op, feed_dict={_a[0]: test_X, y: test_Y}))))
    '''

if __name__ == '__main__':
    '''
    # Loading in the mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels
    trX=trX.astype(np.float64)
    trY=trY.astype(np.float64)
    teX=teX.astype(np.float64)
    teY=teY.astype(np.float64)
    '''
    '''
    for i in range(10):
        for j in range(len(trX[i])):
            if trX[i][j]>=0.99999:
                print(i,'  ',j)
    print(trY.shape)
    '''
    '''
    RBM_hidden_sizes = [500, 200, 50]  # create 4 layers of RBM with size 785-500-200-50
    # Since we are training, set input as training data
    inpX = trX
    # Create list to hold our RBMs
    rbm_list = []
    # Size of inputs is the number of inputs in the training set
    input_size = inpX.shape[1]

    # For each RBM we want to generate
    for i, size in enumerate(RBM_hidden_sizes):
        print('RBM: ', i, ' ', input_size, '->', size)
        rbm_list.append(RBM(input_size, size, 5, 1.0, 100))
        input_size = size

    # For each RBM in our list
    for rbm in rbm_list:
        print('New RBM:')
        # Train a new one
        rbm.train(inpX)
        # Return the output layer
        inpX = rbm.rbm_outpt(inpX)

    nNet = NN(RBM_hidden_sizes, trX, trY, 1.0, 0, 10, 100)
    nNet.load_from_rbms(RBM_hidden_sizes, rbm_list)
    nNet.train()
    '''