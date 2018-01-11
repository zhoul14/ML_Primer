# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class LinearModelReg(object):

    def __init__(self, fdim, ydim, W=None, b=None, visal=False):
        # 数据维度
        self.fdim = fdim
        # 输出维度
        self.ydim = ydim
        # W参数
        self.W = np.matrix(np.zeros([fdim,ydim],dtype=float)) if W is None else W
        # b参数
        self.b = np.zeros([ydim,1],dtype=float) if b is None else b
        # visal 是否可视 W
        self.visal = visal

    def train_GD(self, X, y, lr=0.01, step_num=1):
        for k in range(step_num):
            for i in range(y.shape[2]):
                # print X.shape
                # 更新W
                # W_i+1 = W_i  + 2 * lr * XT *(y - W_i.T * X - b_i )
                # print X[:, :,i], X[:,:,i].T,
                # print "X:",X[:, :,i].shape
                # print "W:",self.W.shape
                # print "------------"
                # print (y[:,:, i] - np.dot(self.W.T, X[:, :,i]) - self.b).shape
                # print "------------"
                self.W += 2 * lr * X[:,:, i] * (y[:,:, i] - np.dot(self.W.T, X[:, :,i]) - self.b)
                # print "W:",self.W
                # print 2 * lr * X[:, i].T * (y[:,:, i] - np.dot(self.W.T, X[:, i]) - self.b)
                # print "bshape:",self.b.shape
                # print (y[:,:, i] - np.dot(self.W.T, X[:, :,i]) - self.b).shape
                # print X[:,:, i].T.shape
                # 更新b
                # b_i+1 = b_i + 2 *  lr * (y - W_i.T * X - b_i )
                # print "------------"
                self.b += 2 * lr * (y[:,:, i] - np.dot(self.W.T, X[:, :,i]) - self.b)
                # print "b:",self.b
                # 如果visal，打印步长
                if self.visal:
                    print "step num:%d, sample num:%d" % (k, i)
                    print "learn rate:%f" % lr
                    print "[param] weight matrix:",self.W
                    print "[param] bias array:",self.b

    def pred(self, X_pred):
        data_lens = X_pred.shape[0]
        y_pred = np.zeros([data_lens,self.ydim],dtype=float)
        for i in range(data_lens):
            y_pred[i] = self.W.T * X_pred[:, i] + self.b
        return y_pred
    def show_params(self):
        print "[param] weight matrix:",self.W
        print "[param] bias array:",self.b


if __name__ == '__main__':
    W = np.matrix([[2],[4]],dtype=float)
    print W.shape
    b = np.array([[2]],dtype=float)
    data_len = 1000
    X_train = np.random.random_sample([2,1,data_len])-0.5
    y_train = np.zeros([1,1,data_len])
    # print X_train
    # print y_train
    for i in range(data_len):
        y_train[:, :,i] = W.T * X_train[:,:, i] + b
    # print "y:",y_train[:,:,0]
    print "----------"
    LM = LinearModelReg(2, 1)
    LM.train_GD(X_train,y_train)
    LM.show_params()
    print "----------"
    LM2 = LinearModelReg(2,1)
    X_train = X_train * 5 + 7
    print X_train
    for i in range(data_len):
        y_train[:, :,i] = W.T * X_train[:,:, i] + b
    print y_train
    LM2.train_GD(X_train,y_train,0.001,9)
    LM2.show_params()

