import numpy as np
from matplotlib import pyplot
import pandas as pd


def sigmoid_func(y):
    return 1 / (1  + np.exp(-1 * y))

def sigmoid_derivate(y):
    return sigmoid_func(y) * (1 - sigmoid_func(y))

def tanh_func(y):
    return (np.exp(y) - np.exp(-1 * y)) / (np.exp(y) + np.exp(-1 * y))

def tanh_derivate(y):
    return 1 - tanh_func(y) * tanh_func(y)

class LMC(object):
    def __init__(self,fdim,ydim,activity_func,visal=True):
        self.W = np.matrix(np.zeros([fdim,ydim],dtype=float))
        self.b = np.matrix(np.zeros([ydim,1],dtype=float))
        self.fdim = fdim
        self.ydim = ydim
        self.activity_func = activity_func
    def train_GD(self,X,y,mini_batch=10,lr=0.01,l2=0.01,activity_func='sigmoid',step_num=1):
        if activity_func == 'sigmoid':
            f = sigmoid_func
            f_d = sigmoid_derivate
        elif activity_func == 'tanh':
            f = tanh_func
            f_d = tanh_derivate
        for k in range(step_num):
            for i in range(y.shape[2]):
                if i % mini_batch == 0:
                    W0 = np.matrix(self.W)
                    b0 = np.matrix(self.b)
                # update weight:W
                self.W += 1 / mini_batch * 2 * lr * f_d(np.dot(W0.T, X[:, :, i]) + b0) * X[:, :, i] \
                          * (y[:, :, i] - f(np.dot(W0.T, X[:, :, i]) + b0)) - l2 * 2 * W0
                # update bias:b
                self.b += 2 * lr * f_d(np.dot(W0.T , X[:, :, i]) + b0) *\
                          (y[:, :, i] - f(np.dot(W0.T, X[:, :, i]) + b0)) * 1 / mini_batch - l2 * 2 * b0
                # print "b:",self.b
                # 如果visal，打印步长
                if self.visal:
                    print "step num:%d, sample num:%d" % (k, i)
                    print "learn rate:%f" % lr
                    print "[param] weight matrix:", self.W
                    print "[param] bias array:", self.b

    def pred(self,X,activity_func):
        sample_len = X.shape[2]
        y_pred = np.zeros([sample_len,self.ydim],dtype=float)
        if activity_func == 'sigmoid':
            f = sigmoid_func
        elif activity_func == 'tanh':
            f = tanh_func
        for i in range(sample_len):
            y_pred[:,:,i] = f(np.dot(self.W.T, X[:, :, i]) + self.b)
        return y_pred