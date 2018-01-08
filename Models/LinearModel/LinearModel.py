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

    def train_GD(self, X, y, lr=0.0008, step_num=3):
        for k in range(step_num):
            for i in range(y.shape[2]):
                # print X.shape
                # 更新W
                # W_i+1 = W_i  + 2 * lr * XT *(y - W_i.T * X - b_i )
                # print X[:, :,i], X[:,:,i].T
                print "X:",X[:, :,i].shape
                print "W:",self.W.shape
                print "------------"
                print (y[:,:, i] - np.dot(self.W.T, X[:, :,i]) - self.b).shape
                print "------------"
                self.W += 2 * lr * X[:,:, i] * (y[:,:, i] - np.dot(self.W.T, X[:, :,i]) - self.b)
                print self.W
                # print 2 * lr * X[:, i].T * (y[:,:, i] - np.dot(self.W.T, X[:, i]) - self.b)
                # print "bshape:",self.b.shape
                # print (y[:,:, i] - np.dot(self.W.T, X[:, :,i]) - self.b).shape
                # print X[:,:, i].T.shape
                # 更新b
                # b_i+1 = b_i + 2 *  lr * (y - W_i.T * X - b_i )
                print "------------"
                self.b += 2 * lr * (y[:,:, i] - np.dot(self.W.T, X[:, :,i]) - self.b)
                print self.b
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


if __name__ == '__main__':
    W = np.matrix([[2],[4]],dtype=float)
    print W.shape
    b = np.array([[2]],dtype=float)
    X_train = np.random.randint(-1,1,[2,1,1000])
    y_train = np.zeros([1,1,1000])
    for i in range(100):
        y_train[:, :,i] = W.T * X_train[:,:, i] + b
    print "y:",y_train[:,:,0]
    print "----------"
    LM = LinearModelReg(2, 1)
    LM.train_GD(X_train,y_train)
