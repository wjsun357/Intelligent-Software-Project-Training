import numpy as np
import tensorflow as tf
import math
from tensorflow.python.framework import ops


def my_sigmoid_def(x, a=5, alpha=0.2):
    # '''
    if x >= a:
        return alpha*(x-a)+1/(1+math.exp(-a))
    elif x <= -a:
        return alpha*(x+a)+1/(1+math.exp(a))
    else:
        return 1/(1+math.exp(-x))
    # '''
    # return 1/(1+math.exp(-x))


def my_sigmoid_grad_def(x, a=5, alpha=0.2):
    # ‘''
    if abs(x) >= a:
        return alpha
    else:
        return math.exp(-x)/pow(1+math.exp(-x), 2)
    # '''
    # return math.exp(-x)/pow(1+math.exp(-x),2)


my_sigmoid_np = np.vectorize(my_sigmoid_def)
my_sigmoid_grad_np = np.vectorize(my_sigmoid_grad_def)
my_sigmoid_np_64 = lambda x: my_sigmoid_np(x).astype(np.float64)
my_sigmoid_grad_np_64 = lambda x: my_sigmoid_grad_np(x).astype(np.float64)


def my_sigmoid_grad_tf(x, name=None):
    with ops.name_scope(name, "my_sigmoid_grad_tf", [x]) as name:
        y = tf.py_func(my_sigmoid_grad_np_64,
                       [x],
                       [tf.float64],
                       name=name,
                       stateful=False)
        return y[0]


def my_py_func(func, inp, Tout, stateful=False, name=None, my_grad_func=None):
    # Need to generate a unique name to avoid duplicates:
    random_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(random_name)(my_grad_func)  # see _my_sigmoid_grad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": random_name, "PyFuncStateless": random_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def _my_sigmoid_grad(op, pre_grad):
    x = op.inputs[0]
    cur_grad = my_sigmoid_grad_tf(x)
    next_grad = pre_grad * cur_grad
    return next_grad


def my_sigmoid_tf(x, name=None):
    with ops.name_scope(name, "my_sigmoid_tf", [x]) as name:
        y = my_py_func(my_sigmoid_np_64,
                       [x],
                       [tf.float64],
                       stateful=False,
                       name=name,
                       my_grad_func=_my_sigmoid_grad)  # <-- here's the call to the gradient
        return y[0]


