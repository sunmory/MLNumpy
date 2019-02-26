# -*- coding: utf-8 -*-

import numpy as np


class SvmModel:
    def __init__(self, training_data, label, C, max_iter, tole, kernal_parameter):
        self.training_data = training_data
        self.label = label
        self.C = C
        self.max_iter = max_iter
        self.tole = tole
        self.kernal_parameter = kernal_parameter
        self.m, self.n = training_data.shape
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.K = self.kernal()
        self.b = 0

    def kernal(self):
        m = self.training_data.shape[0]
        K = np.mat(np.zeros((m, m)))
        if self.kernal_parameter[0] == 'lin':
            K = np.dot(self.training_data, self.training_data.T)
        elif self.kernal_parameter[0] == 'rbf':
            for i in range(m):
                for j in range(m):
                    result1 = self.training_data[i, :] - self.training_data[j, :]
                    # print training_data[i,:]
                    K[i, j] = np.exp((result1 * result1.T) / (-1 * float(self.kernal_parameter[1] ** 2)))
        return K

    def cal_E(self, i):
        """
        计算Ei
        Ei = g(xi) - yi
        :param i: 出的待更新的乘子的索引
        :return:
        """
        K_i = self.K[:, i]  # 取出K(xj, xi)(j = 1 ~ n)
        fx = float(np.multiply(self.alphas, self.label).T * K_i + self.b)  # 计算g(xi)
        E = fx - self.label[i]  # 计算Ei
        return E

    def select_j(self, alphas_index1):
        """
        选择更新幅度最大的参数为a2
        :param alphas_index1: 第一个参数的索引
        :return:
        """
        m = self.alphas.shape[0]
        max_distance = 0
        alphas_index2 = 0
        E_i = self.cal_E(alphas_index1)
        for K in range(m):
            E_K = self.cal_E(K)
            distance = np.abs(E_i - E_K)
            if distance > max_distance:
                max_distance = distance
                E_j = E_K
                alphas_index2 = K
        return (E_i, E_j, alphas_index2)

    def update_alphas(self, E_i, E_j, alphas_index1, alphas_index2):
        """
        更新参数a与b
        :param E_i: a1的E
        :param E_j: a2的E
        :param alphas_index1: 第一个参数的索引
        :param alphas_index2: 第二个参数的索引
        :return:
        """
        distance_value = float(E_i - E_j)
        eta = self.K[alphas_index1, alphas_index1] + self.K[alphas_index2, alphas_index2] - 2 * self.K[
            alphas_index1, alphas_index2]
        alphas2_new = self.alphas[alphas_index2] + self.label[alphas_index2] * distance_value / float(eta)
        alphas2_new = self.alphas_range(alphas_index1, alphas_index2, alphas2_new)
        alphas1_new = self.alphas[alphas_index1] + self.label[alphas_index1] * self.label[alphas_index2] * (
                self.alphas[alphas_index2] - alphas2_new)
        b_new1 = - E_i - self.label[alphas_index1] * self.K[alphas_index1, alphas_index1] * (
                alphas1_new - self.alphas[alphas_index1]) - \
                 self.label[alphas_index2] * self.K[alphas_index2, alphas_index1] * (
                         alphas2_new - self.alphas[alphas_index2]) + self.b
        b_new2 = - E_j - self.label[alphas_index1] * self.K[alphas_index1, alphas_index2] * (
                alphas1_new - self.alphas[alphas_index1]) - \
                 self.label[alphas_index2] * self.K[alphas_index2, alphas_index2] * (
                         alphas2_new - self.alphas[alphas_index2]) + self.b

        # 从b1与b2中选择合适的b

        if (alphas1_new > 0) and (alphas1_new < self.C):
            b_new = b_new1
        elif (alphas2_new > 0) and (alphas2_new < self.C):
            b_new = b_new2
        else:
            b_new = float(b_new1 + b_new2) / 2

        return (alphas1_new, alphas2_new, b_new)

    def alphas_range(self, alphas_index1, alphas_index2, alphas_new):
        """
        计算a的范围，将计算得到的a的值限制在正确的范围内
        :param alphas_index1: a1的索引
        :param alphas_index2: a2的索引
        :param alphas_new: 计算得到的新的a2的值，送入函数中对范围进行限制
        :return:
        """
        if self.label[alphas_index1] != self.label[alphas_index2]:
            H = min(self.C, self.C + self.alphas[alphas_index2] - self.alphas[alphas_index1])
            L = max(0, self.alphas[alphas_index2] - self.alphas[alphas_index1])
        else:
            H = min(self.C, self.alphas[alphas_index2] + self.alphas[alphas_index1])
            L = max(0, self.alphas[alphas_index2] + self.alphas[alphas_index1] - self.C)
        alphas_new = min(alphas_new, H)
        alphas_new = max(alphas_new, L)
        return alphas_new

    def smo(self):
        """
        smo算法
        :return:
        """
        iter_num = 0
        unfit_alphas_num = 1

        # 当迭代次数小于最大迭代次数或上次循环中仍存在有价值的拉格朗日乘子更新的时候继续进行更新

        while (iter_num < self.max_iter) and (unfit_alphas_num != 0):
            print("inter num:%d" % iter_num)
            # print self.alphas
            unfit_alphas_num = 0

            # KKT条件：
            # 先遍历 a > 0 且 a < C 的参数，找出不满足KKT条件的为a1
            # 若无，则再遍历 a == 0 或 a == C 的参数，找出不满足KKT条件的为a1
            # 找到第一个不满足KKT条件的参数a1后，第二个参数a2应选择使a2能进行最大幅度更新的，即使得|E1 - E2|最大

            for index, x in enumerate(self.alphas):
                if (x > 0) and (x < self.C) and \
                        (np.abs(self.label[index] * self.cal_E(index)) > self.tole):
                    E_i, E_j, index2 = self.select_j(index)  # 选择更新幅度最大的参数为a2
                    alphas1_old = self.alphas[index].copy()  # 重点，非copy的变量传值实际为指针，copy才会实际再分配一个内存来存储数值
                    self.alphas[index], self.alphas[index2], self.b = self.update_alphas(E_i, E_j, index, index2)
                    if (np.abs(self.alphas[index] - alphas1_old) > 0.000001):
                        unfit_alphas_num = unfit_alphas_num + 1
                # print("here")
            for index, x in enumerate(self.alphas):
                # print (-(self.label[index] * self.cal_E(index))
                if ((x == 0) and (-(self.label[index] * self.cal_E(index)) > self.tole)) \
                        or ((x == self.C) and (self.label[index] * self.cal_E(index)) > self.tole):
                    E_i, E_j, index2 = self.select_j(index)
                    alphas1_old = self.alphas[index].copy()  # 重点，非copy的变量传值实际为指针
                    self.alphas[index], self.alphas[index2], self.b = self.update_alphas(E_i, E_j, index, index2)
                    if (np.abs(self.alphas[index] - alphas1_old) > 0.000000001):
                        unfit_alphas_num = unfit_alphas_num + 1

            iter_num = iter_num + 1
        return

    def predict(self, text):
        """
        对输入样本进行预测分析
        :param text: 输入的待预测的样本
        :return: 正负类预测结果
        """

        svindex = np.nonzero(self.alphas)[0]
        svs = self.training_data[svindex]
        svs_alphas = self.alphas[svindex]
        svs_label = self.label[svindex]
        svs_num = svs.shape[0]
        K = np.mat(np.zeros((svs_num, 1)))
        for i in range(svs_num):
            result1 = svs[i, :] - text
            K[i] = np.exp((result1 * result1.T) / (-1 * float(self.kernal_parameter[1] ** 2)))
        category = np.sign(np.multiply(svs_alphas, svs_label).T * K + self.b)
        return int(category)


def img2vector(file_name):
    return_vect = np.zeros((1, 1024))
    fr = open(file_name)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def load_images(dirName):
    from os import listdir
    labels = []
    train_file_list = listdir(dirName)  # load the training set
    m = len(train_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name_str = train_file_list[i]
        file_str = file_name_str.split('.')[0]  # take off .txt
        class_num_str = int(file_str.split('_')[0])
        if class_num_str == 9:
            labels.append(-1)
        else:
            labels.append(1)
        training_mat[i, :] = img2vector('%s/%s' % (dirName, file_name_str))
    return np.mat(training_mat), np.mat(labels).T


def svm_test(C, max_iter, tole, kernal_parameter):
    """
    运行SVM算法
    :param C: 惩罚因子，用于限制松弛变量的大小
    :param max_iter: 最大迭代次数，指更新不符合KKT条件的拉格朗日乘子的最大次数
    :param tole: 松弛变量，防止过拟合
    :param kernal_parameter: 核函数因子，包括使用哪种核函数与相应的核函数参数（如：rbf-高斯核函数 高斯核参数为 22 ）
    :return: None
    """

    training_data, label = load_images("..\\Data\\digits\\trainingDigits")
    svm = SvmModel(training_data, label, C, max_iter, tole, kernal_parameter)
    svm.smo()
    sample_num = training_data.shape[0]
    correct_num = 0
    for i in range(sample_num):
        if svm.predict(training_data[i]) == label[i, 0]:
            correct_num = correct_num + 1
    accuracy_rate = float(correct_num) / float(sample_num)
    print('train error rate is %f' % (1 - accuracy_rate))

    test_data, label = load_images("..\\Data\\digits\\testDigits")
    sample_num = test_data.shape[0]
    correct_num = 0
    for i in range(sample_num):
        if svm.predict(test_data[i]) == label[i, 0]:
            correct_num = correct_num + 1
    accuracy_rate = float(correct_num) / float(sample_num)
    print('test error rate is %f' % (1 - accuracy_rate))


if __name__ == '__main__':
    svm_test(1, 200, 0.000001, ('rbf', 10))