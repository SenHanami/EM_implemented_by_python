import sklearn
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torchvision.datasets as datasets
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import os
import data_getter
import pickle
import copy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tqdm import tqdm
import math
import time
from itertools import product
from sklearn.cluster import KMeans
from visualization import visualization_2d as vs2


# 纯EM,均值采用随机抽样生成
class Em():
    def __init__(self, dim, category_num):
        '''
        :param dim: the dimension of  Gaussian distribution
        :param category_num: the numbers of category
        '''

        self.dim = dim
        self.category_num = category_num
        self.u = None
        self.covariance_matrix = None
        self.w = None  # 各类别的概率
        self.p_matrix = None  # 后验概率密度矩阵
        self.temp1 = None

    def p_getter_1(self, x):
        '''
        获得p(x)
        :param sample:sample
        :return:p(x)
        '''
        return np.array([self.p_getter_2(x, i)[1] for i in range(self.category_num)]).reshape((1, -1)).dot(
            self.w.reshape((-1, 1)))

    def p_getter_2(self, x, ci):
        '''
        获得p(category)*p(x|category)
        :param x: sample
        :param ci: category_index [start from 0]
        :return: p(category)*p(x|category) ,p(x|category)
        '''

        rv = multivariate_normal(self.u[ci], self.covariance_matrix[ci])
        p_x_con = rv.pdf(x)
        return self.w[ci] * p_x_con, p_x_con

    # 生成贝叶斯后验概率矩阵,并更新参数.aij：第j个值属于第i类的概率密度
    def ite(self, x_train):
        temp = np.zeros_like(self.p_matrix).T
        for j in range(self.category_num):
            temp[:, j] = multivariate_normal.pdf(x_train, self.u[j], self.covariance_matrix[j])
        temp = np.multiply(temp, self.w).T
        self.temp1 = np.sum(temp, axis=0)
        for i in range(x_train.shape[0]):
            p_x = np.sum(temp[:, i])
            for j in range(self.category_num):
                self.p_matrix[j, i] = temp[j, i] / p_x

        for i in range(self.category_num):
            self.w[i] = self.p_matrix[i, :].sum() / x_train.shape[0]
            self.u[i] = np.average(x_train, axis=0, weights=(self.p_matrix[i] / (self.p_matrix[i].sum())))
            a = self.p_matrix[i].sum()
            ls = []
            for j in range(x_train.shape[0]):
                v = np.matrix(x_train[j] - self.u[i])
                ls.append(self.p_matrix[i, j] * np.dot(v.T, v) / a)
            self.covariance_matrix[i] = sum(ls)

    def initial_cov_matrix_getter(self):
        cov_matrix = np.ones((self.category_num, self.dim, self.dim))
        for i in range(self.category_num):
            cov_matrix[i] = 500 * (np.ones((self.dim, self.dim)) + np.eye(self.dim))
        return cov_matrix

    def fit_predict(self, x_train):
        # 生成初始的参数
        self.u = np.zeros((self.category_num, x_train.shape[1]))
        for i in range(self.category_num):
            np.random.shuffle(x_train)
            self.u[i] = np.average(x_train[0:200], axis=0)

        self.w = np.ones((self.category_num)) * (1 / self.category_num)
        self.covariance_matrix = self.initial_cov_matrix_getter()
        self.p_matrix = np.zeros((self.category_num, x_train.shape[0]))

        p1 = 0
        while True:
            self.ite(x_train)
            p0 = p1
            p1 = np.sum(np.log(np.array([self.temp1[i] for i in range(x_train.shape[0])])))
            print('loglike:{} \n'.format(p1))
            if abs(p1 - p0) < 1:
                break

        p_matrix = torch.tensor(self.p_matrix)
        idxs = p_matrix.max(0, keepdim=True)[-1]
        return idxs.numpy().ravel()