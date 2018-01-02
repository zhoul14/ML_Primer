import numpy as np
import scipy
import pandas as pd
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class base_data(object):
    x = np.random.rand(100)
    y = np.random.rand(100)
    data_len = 100
    dim = 1

    def __init__(self, data_len, dim):
        self.data_len = data_len
        self.dim = dim


class sin_data(base_data):
    def __init__(self, data_len, dim, alpha):
        base_data.__init__(self, data_len, dim)
        self.x = np.sort(np.random.rand(data_len))
        self.y = np.sin(self.x * 2 * np.pi * alpha)


class line_data(base_data):
    def __init__(self, data_len, dim, w, b):
        base_data.__init__(self, data_len, dim)
        self.x = np.sort(np.random.rand(data_len))
        self.y = np.array(w * self.x + b)


class GMM_data(base_data):
    def __init__(self, data_len, dim=1, mu=0, sigma=1, mixture=1, aplha=1):
        base_data.__init__(self, data_len, dim)
        self.y = np.zeros([data_len, 1], dtype= float)
        if dim > 1:
            self.x = np.matrix(np.random.rand(data_len, dim))
            for i in range(mixture):
                self.y += aplha[i] * 1 / np.sqrt(2 * np.pi * np.linalg.det(sigma[i])) * np.exp(
                    -(self.x - mu[i]) * np.linalg.inv(sigma[i]) * (self.x - mu[i]))
        else:
            self.x = np.random.rand(data_len, dim)
            for i in range(mixture):
                self.y += aplha[i] * 1 / np.sqrt(2 * np.pi * sigma[i]) * np.exp(
                    -(self.x - mu[i]) * (1 / sigma[i]) * (self.x - mu[i]))


class norm_data(GMM_data):
    def __init__(self, data_len, dim=1, mu=0, sigma=1):
        GMM_data.__init__(self, data_len, dim, mu, sigma)


class RGS_datas(object):
    x = np.random.rand(100)
    y = np.random.rand(100)
    data_len = 100
    dim = 2

    def __init__(self, data_len, dim):
        self.data_len = data_len
        self.dim = dim
        self.linear = line_data(100,1,2,4)
        self.GMM = GMM_data(100,2,[0],[2],1,[1])
        self.sin = sin_data(100,1,1)

    def set_linear(self,w,b):
        self.linear=line_data(self.data_len,self.dim,w,b)

    def get_linear_data(self):
        return self.linear

    def set_sin(self,alpha):
        self.sin = sin_data(data_len=self.data_len,dim=self.dim,alpha=alpha)

    def get_sin_data(self):
        return self.sin

    # def init_sin(self):
    #     self.sin =
    #     # self.type = type
    #     # self.x = np.random.rand(self.data_len)
    #     if type == 'sin':
    #         self.y = np.sin(self.x * 2 * np.pi)
    #     if type == 'gauss':
    #         self.x = np.random.multivariate_normal(p1, p2, p3)
    #     if type == 'GMM':
    #         self.x = np.random.multivariate_normal(p1, p2, p3) +  np.random.multivariate_normal(p1, p2, p3, p4)
    #         self.y =

    def show_data(self,args="b-"):
        if self.dim == 1 :
            plt.plot(self.sin.x,self.sin.y,args)
            plt.show()
        if self.dim == 2:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(self.x, self.y, self.z, rstride=1, cstride=1, cmap='rainbow')

r = RGS_datas(100,2)
r.show_data()