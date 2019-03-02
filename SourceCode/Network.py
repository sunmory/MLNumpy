# -*- coding: utf-8 -*-

import pickle as pkl
import random
import numpy as np


def load_data(path='..\\data\\mnist\\mnist.pkl'):
    """
    读取样本数据
    :param path: 数据路径
    :return:
    """
    f = open(path, 'rb')
    training_data, validation_data, test_data = pkl.load(f)
    training_data, validation_data, test_data = data_wrapper(training_data), data_wrapper(
        validation_data), data_wrapper(test_data)
    return (training_data, validation_data, test_data)


def data_wrapper(data):
    """
    对数据进行封装：
    - 将样本特征（像素）变为 784 行 1 列的矩阵（numpy中数组可作为向量来与矩阵相乘）
    - 将标签化为神经网络的输出形式（即10*1的向量）
    - 将特征与标签组合为一个元组，便于后续计算
    :return:
    """
    data_x = [np.reshape(x, (784, 1)) for x in data[0]]
    data_y = [change_label(y) for y in data[1]]
    data = zip(data_x, data_y)
    return data


def change_label(n):
    """
    将标签化为10*1的神经网络的输出向量
    :param n: 样本标签值
    :return:
    """
    y = np.zeros((10, 1))
    y[n] = 1.0
    return y


class Network(object):

    def __init__(self, size, eta):
        """
        :param size: size = (num1, num2, num3,...)表示每层的单元数，用于设置神经网络大小
        :param eta: 学习速率
        """
        self.size = size
        self.eta = float(eta)
        self.layers = len(size)
        # numpy中数组可作为向量来与矩阵相乘，self.bias = [np.mat(np.random.randn(y , 1)) for y in self.size[1: ]] 多余
        self.bias = [np.mat(np.random.randn(y, 1)) for y in self.size[1:]]
        self.weight = [np.mat(np.random.randn(x, y)) for x, y in zip(self.size[: -1], self.size[1:])]

    def forward(self, data_x):
        """
        前向传播算法计算一个样本的各个神经单元的中间结果 z 和输出结果 a
        z 向量个数为网络层数 - 1
        a 向量个数为网络层数
        :param data_x:
        :return:
        """
        activations = [data_x]
        zs = []
        a = data_x
        for w, b in zip(self.weight, self.bias):
            z = np.dot(w.T, a) + b
            a = self.sigmod(z)
            zs.append(z)
            activations.append(a)
        return (zs, activations)

    def sigmod(self, z):
        """
        sigmod函数
        """
        return (1.0 / (1.0 + np.exp(-z)))

    def sgd(self, training_data, validation_data, mini_batch_size, max_round):
        """
        随机梯度下降算法
        目标：
        - 在每次循环中将随机排序后的训练样本划分为一个个小的批次
        - 使用每个批次的样本进行一次神经网络的参数更新
        :param training_data: 数据集
        :param validation_data: 验证集
        :param mini_batch_size: 为每次更新所使用的样本数
        :param max_round: 最大迭代次数
        :return:
        """
        data_size = len(training_data)
        for round_num in range(max_round):
            random.shuffle(training_data)  # 重点，不赋值 training_data = random.shuffle(training_data)
            '''
            python中可迭代元素的索引超出范围后会自动在取到最后的数值后停止
            '''
            data_set = [training_data[k: k + mini_batch_size] for k in range(0, data_size, mini_batch_size)]
            for data_list in data_set:
                self.network_update(data_list)
            print('round_num: {0} -- training accuracy rate: {1} '.format(round_num, self.evaluate(training_data)))
            print('round_num: {0} -- test accuracy rate: {1} '.format(round_num, self.evaluate(validation_data)))
        return

    def network_update(self, data_list):
        """
        参数更新算法
        目标：
        - 首先初始化偏置与参数矩阵为零用于累加每个样本对参数的偏导数值
        - 使用学习率，批次的样本数与批次样本的参数偏导之和对神将网络的参数进行一次更新
        :param data_list: 用于进行一次更新的样本批次
        :return:
        """
        n = len(data_list)
        new_bias = [np.mat(np.zeros((y, 1))) for y in self.size[1:]]
        new_weight = [np.mat(np.zeros((x, y))) for x, y in zip(self.size[: -1], self.size[1:])]
        for single_data in data_list:
            update_bias, update_weight = self.backpro(single_data)
            new_bias = [bias1 + bias2 for bias1, bias2 in zip(new_bias, update_bias)]
            new_weight = [weight1 + weight2 for weight1, weight2 in zip(new_weight, update_weight)]
        self.bias = [b - (self.eta / n) * n_b for b, n_b in zip(self.bias, new_bias)]
        self.weight = [w - (self.eta / n) * n_w for w, n_w in zip(self.weight, new_weight)]
        return

    def backpro(self, single_data):
        """
        后向传播算法
        目标：
        - 后向传播算法计算各个单元的误差项（计算参数偏导的必要部分，需要由后向前推导）
        - 使用目标层各个单元的输出项与后一层各个单元的误差项计算参数的偏导（由于偏置单元的目标单元值为1，所以偏置参数数值直接为误差项）
        :param single_data: 一个单一的样本
        :return:
        """
        data_x, label = single_data[0], single_data[1]
        update_bias = [np.zeros(b.shape) for b in self.bias]
        update_weight = [np.zeros(w.shape) for w in self.weight]
        zs, activations = self.forward(data_x)

        # 初始化误差项（输出单元的输出值与标签值之差）

        delt = activations[-1] - label
        update_bias[-1] = delt
        update_weight[-1] = (np.dot(activations[-2], delt.T))
        for l in range(2, self.layers):
            # print l
            delt = np.multiply(np.dot(self.weight[-l + 1], delt), self.sigmod_prime(zs[-l]))
            update_bias[-l] = delt
            update_weight[-l] = np.dot(activations[-l - 1], delt.T)
        return (update_bias, update_weight)

    def sigmod_prime(self, z):
        """
        sigmod函数求导（sigmod函数有特殊的求导特性，其导数可以由自身表示）
        :return:
        """
        return np.multiply(self.sigmod(z), (1 - self.sigmod(z)))

    def evaluate(self, validation_data):
        """
        评估函数
        - 神经网络通过输出单元中最大的数值的数组中序号来表示判断的数字结果
        :param validation_data: 验证集
        :return:
        """
        total = len(validation_data)
        correct = 0
        for single_data in validation_data:
            data_x, data_y = single_data[0], single_data[1]
            zs, activations = self.forward(single_data[0])
            if np.argmax(activations[-1]) == np.argmax(data_y):
                correct = correct + 1
        return float(correct) / float(total)


if __name__ == '__main__':
    training_data, validation_data, test_data = load_data()
    bp_network = Network((784, 30, 10), 3)
    bp_network.sgd(training_data, validation_data, 10, 30)
