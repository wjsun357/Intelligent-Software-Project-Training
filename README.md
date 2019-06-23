# 智能软件项目实训
该实训文档主要包括了6个文件夹<br>
`Code`-源代码及实验结果<br>
`Data`-轴承故障原始数据及降噪数据<br>
`Paper`-最终提交的小论文<br>
`References`-部分参考文献<br>
`Report`-word与ppt报告文档<br>
`Request`-实训要求<br>
<br><br>
在最终的实验过程中，主要用到的代码是[DBN.py](https://github.com/wjsunscut/Intelligent-Software-Project-Training/blob/master/Code/DBN.py)（DBN实现）、[SSDBN.py](https://github.com/wjsunscut/Intelligent-Software-Project-Training/blob/master/Code/SSDBN.py)（SSDBN实现）、[draw.py](https://github.com/wjsunscut/Intelligent-Software-Project-Training/blob/master/Code/draw.py)（绘制对比图，使用[Code/result](https://github.com/wjsunscut/Intelligent-Software-Project-Training/tree/master/Code/result)中的数据）、[isigmoid.py](https://github.com/wjsunscut/Intelligent-Software-Project-Training/blob/master/Code/isigmoid.py)（Isigmoid实现）、[main.py](https://github.com/wjsunscut/Intelligent-Software-Project-Training/blob/master/Code/main.py)（主程序）、[preprocessing.m](https://github.com/wjsunscut/Intelligent-Software-Project-Training/blob/master/Code/preprocessing.m)（降噪）<br>
<br><br>
DBN中RBM的初始化函数`__init__(self, input_size, output_size, epoches, learning_rate, batchsize)`中，input_size为可见层单元数，output_size为隐含层单元数，epoches为循环次数，learning_rate为学习率，batchsize为批尺寸。`train(self, X)`中，X为输入数据。<br>
NN使用`load_from_rbms(self, dbn_sizes, rbm_list)`获取预训练参数。`train(self, test_X, test_Y)`中，test_X为输入，test_Y为标签。<br>
SSDBN的SSRBM的初始化函数`__init__(self, input_size, output_size, epoches, learning_rate, batchsize, proportion)`相较RBM多了一个proportion，用以调整比例。`train(self, X, S)`中的S为u的输入数据。<br>
<br><br>
RBM的设置参数为epoches=100，learning_rate=0.3，batchsize=30，SSRBM的参数为proportion=1，Isigmoid的参数为a=5，alpha=0.2。NN的参数为epoches=1000，learning_rate=1，batchsize=30，前2/3动量为0.9，后1/3动量为0.5。<br>
<br><br>
主要的源代码在[Code.zip](https://github.com/wjsunscut/Intelligent-Software-Project-Training/blob/master/Code/Code.zip)中。<br>
直接运行[main.py](https://github.com/wjsunscut/Intelligent-Software-Project-Training/blob/master/Code/main.py)即可（可根据需要调整代码以分别运行DBN和SSDBN）。
