"""
Feature selection considering shape = 2
working version with my equations
"""

import math
from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot
from scipy.special import digamma
from scipy.special import gamma
from sklearn.cluster import KMeans
from scipy import special
import mpmath

import sys


def lowerbound_first_dir(rnk, x, shape, mk, s0, e_precision_):
    x_mu = np.abs(x.reshape(-1, 1) - mk)
    p11 = np.power(x_mu, shape)
    p12 = np.log(x_mu)
    p1 = p11 * p12 * (s0 - e_precision_)
    p2 = -(1 / np.power(shape, 2)) * np.log(np.abs(e_precision_)) + 1 / shape - 1 / (2 * gamma(1 / shape)) * digamma(
        1 / shape)
    p3 = e_precision_ * np.power(x_mu, shape) * np.log(x_mu)
    return rnk * (p1 + p2 + p3)


def lowerbound_second_dir(rnk, x, shape, mk, s0, e_precision_):
    x_mu = np.abs(x.reshape(-1, 1) - mk)
    p1 = 2 * np.power(x_mu, shape) * np.log(x_mu) * (s0 - e_precision_)

    p2 = (2 / np.power(shape, 3)) * np.log(np.abs(e_precision_)) - 1 / np.power(shape, 2) + 1 / (
            2 * np.power(gamma(1 / shape), 2)) * np.power(digamma(1 / shape), 2) - 1 / (
                 2 * gamma(1 / shape)) * special.polygamma(1, 1 / shape)

    p3 = e_precision_ * 2 * np.power(x_mu, shape) * np.log(x_mu)
    return rnk * (p1 + p2 + p3)


def initilization(k, x):
    # init
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x.reshape(-1, 1))

    means = kmeans.cluster_centers_
    clustered_img = defaultdict(list)

    labels = kmeans.labels_

    for v, k in zip(x, labels): clustered_img[k].append(v)

    variance = []
    for i in clustered_img.keys():
        variance.append(np.var(np.asarray(clustered_img.get(i))))
    variance = np.asarray(variance)
    # precision = 1 / variance
    weights = []

    for i in clustered_img.keys():
        weights.append(len(clustered_img.get(i)) / len(x))

    weights = np.asarray(weights)
    # resp = np.hstack((labels.reshape(-1,1), labels_1.reshape(-1, 1)))
    return (means.reshape(-1), variance, weights, labels)


# E[ln pik]
def e_ln_pi_k(gama0, Nk):
    gammak = gama0 + Nk
    return digamma(gammak) - digamma(gammak.sum())


# E[ln(T)]
def e_ln_precision(alpha, beta):
    return digamma(alpha) - np.log(beta)


def e_x_mean_lamnda_2(mk, x, s0):
    return 1 / s0 + np.power(x.reshape(-1, 1) - mk, 2)


def e_mean_n(sk, mk, shape, k):
    """
    E(|mean^shape|)
    """

    temp = []
    # shape = 2
    for cluster in range(k):
        # print("HYP1F1: ",-shape / 2, 1 / 2, -1 / 2 * mk[cluster] ** 2 * sk[0])
        p = (1 / math.sqrt(abs(sk[cluster]))) ** shape * 2 ** (shape / 2) * gamma((1 + shape) / 2) / math.sqrt(
            math.pi) * mpmath.hyp1f1(-shape / 2, 1 / 2, -1 / 2 * mk[cluster] ** 2 * sk[cluster])
        temp.append(p)
    return np.asarray(temp).reshape(-1, 1)


# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSamplelessres.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/starfish.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testimg.png"
input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample_copy.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testimg.png"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/circle.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/birds2.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/demo.png"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/crow copy.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/starfish.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/face.jpg"

x = cv2.imread(input_img_path, 0)
# x = (x-x.min())/(x.max()-x.min())
N = len(x)
o_shape = x.shape
x = x.reshape(-1)
x = x + 0.0
k = 2

alpha0 = np.asarray([x.mean() ** 2 / x.var() for _ in range(k)])
beta0 = np.asarray([x.mean() / i for i in alpha0])
# beta0 = np.asarray([(1e-20) * 1 / i for i in alpha0])
# shape = 2
shape = np.asarray([2 for _ in range(k)])
m0, s0, gama0, resp = initilization(k, x)

# s0 = 1/s0

# pyplot.subplot(3, 1, 2)
pyplot.imshow(resp.reshape(o_shape))
pyplot.show()
temp = np.zeros((len(x), k))
z = []
import pandas

for i, j in zip(resp, temp):
    j[i] = 1
    z.append(j)

z = np.asarray(z)
# z = np.array([np.random.dirichlet(np.ones(k)) for _ in range(len(x))])
rnk = np.exp(z) / np.reshape(np.exp(z).sum(axis=1), (-1, 1))
Nk = rnk.sum(axis=0)
sk = s0 + 2 * Nk * (alpha0 / beta0)

test_mk_num = z.copy()
test_mk_den = z.copy()

mk = m0.copy()

e_precision_ = alpha0 / beta0

for cluster in range(k):
    # np.multiply(rnk * e_precision_[cluster] * shape[cluster] * x.reshape(-1, 1) * shape[cluster], np.where(x>mk[cluster]))

    # test_mk_num = rnk * e_precision_[cluster] * shape[cluster] * x.reshape(-1, 1) * shape[cluster]
    #
    # test_mk_den = rnk * e_precision_[cluster] * np.power(x, shape[cluster]).reshape(-1, 1) * shape[cluster] * \
    #               (shape[cluster]-1)/(2*np.power(x, 2))

    for i in range(len(x)):
        if x[i] > mk[cluster]:
            test_mk_num[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * shape[cluster] * abs(
                x[i] ** shape[cluster]) / (
                                              2 * x[i])

            test_mk_den[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * abs(x[i]) ** shape[cluster] * shape[
                cluster] * (shape[cluster] - 1) / (2 * x[i] ** 2)
        else:

            test_mk_den[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * mk[cluster] ** (shape[cluster] - 2)
            if test_mk_den[i, cluster] <= 0:
                test_mk_den[i, cluster] = 0

            term0 = rnk[i][cluster] * e_precision_[cluster] * shape[cluster] / 2 * mk[cluster] ** (
                    shape[cluster] - 2) * x[i]

            term1 = rnk[i][cluster] * e_precision_[cluster] * shape[cluster] / 4 * (shape[cluster] - 1) * mk[
                cluster] ** (shape[cluster] - 3) * x[i] ** 2

            if term1 <= 0:
                term1 = 0

            if term0 <= 0:
                term0 = 0

            test_mk_num[i, cluster] = term1 + term0

    numerator = test_mk_num.sum(axis=0)[cluster] + s0[cluster] * m0[cluster] / 2
    sk[cluster] = test_mk_den.sum(axis=0)[cluster] + s0[cluster] / 2
    mk[cluster] = numerator / sk[cluster]

gammak = gama0 + Nk
alphak = Nk / 2 + alpha0 - 1
betak = beta0 + (rnk * e_x_mean_lamnda_2(mk, x, sk)).sum(axis=0)
e_ln_pi = e_ln_pi_k(gammak, Nk)
e_ln_precision_ = digamma(alphak) - np.log(betak)
e_precision_ = alphak / betak
e_x_mean_lambda_ = z.copy()

# e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, sk)

for cluster in range(k):
    for i in range(len(x)):
        if x[i] > mk[cluster]:
            if x[i] != 0:
                e_x_mean_lambda_[i, cluster] = x[i] ** shape[cluster] - shape[cluster] * x[i] ** shape[cluster] / x[i] * \
                                               mk[
                                                   cluster] + shape[cluster] / 2 * (
                                                       shape[cluster] - 1) * x[i] ** shape[cluster] / x[i] ** 2 * (
                                                       1 / sk[cluster] + mk[cluster] ** 2)

        else:

            t1 = -shape[cluster] * x[i] * e_mean_n(sk, mk, shape[cluster] - 1, k)
            e_mean2 = e_mean_n(sk, mk, shape[cluster] - 2, k)
            t2 = [0, 0]

            if shape[cluster] > 1:
                t2 = shape[cluster] / 2 * (shape[cluster] - 1) * x[i] ** 2 * e_mean2

            e_x_mean_lambda_[i, cluster] = abs(e_mean_n(sk, mk, shape[cluster], k)[cluster] + t1[cluster] + t2[cluster])

            # e_x_mean_lambda_[i, cluster] = abs(e_x_mean_lambda_[i, cluster])
count = 0

fik = rnk
fik_1 = rnk
phi = rnk.sum(axis=0) / len(rnk)

f_mean = mk
f_precision = sk

while (count < 50):
    for i in range(k):
        p1 = (1 / shape[i]) * e_ln_precision_[i] + np.log(shape[i]) - np.log(2 * gamma(1 / shape[i]))
        p2 = (p1 - (e_precision_[i] * e_x_mean_lambda_[:, i]).reshape(-1, 1)).reshape(-1)

        fik[:, i] = rnk[:, i] * (p2 + np.log(phi[i]))

        f_mean[i] = (fik[:, i] * x).sum(axis=0) / fik[:, i].sum(axis=0)

        f_precision[i] = (fik[:, i] * (x - f_mean[i]) ** shape[i]).sum(axis=0) / fik[:, i].sum()

        # fik_1[:, i] = np.log(shape[i]) + 1/2*np.log(f_precision[i]) - np.log(2*gamma(1/shape[i])) - f_precision[i] *(x - f_mean[i])**shape[i]
        #
        # second_term = fik_1[:, i] * (fik_1[:,i] + np.log(1 - phi[i]))
        p3 = e_ln_pi[i] + p2 * fik[:, i]
        z[:, i] = p3


    phi = fik.sum(axis=0) / len(fik)

    # z[:, i] = (p1 - (e_precision_[i] * e_x_mean_lambda_[:, i]).reshape(-1, 1)).reshape(-1)

    # b = z.max(axis=0)
    # b1 = np.exp(z - b)
    # rnk = b1/b1.sum(axis=0)

    rnk = np.exp(z) / np.reshape(np.exp(z).sum(axis=1), (-1, 1))
    Nk = rnk.sum(axis=0)

    alphak = (rnk * fik).sum(axis=0) / 2 + alpha0 - 1
    # alphak = Nk / 2 + alpha0 - 1

    # betak = beta0 + (rnk * e_x_mean_lambda_).sum(axis=0)
    betak = beta0 + (rnk * fik * e_x_mean_lambda_).sum(axis=0)

    for cluster in range(k):
        for i in range(len(x)):
            if x[i] > mk[cluster]:
                test_mk_num[i, cluster] = fik[i][cluster] * rnk[i][cluster] * e_precision_[cluster] * shape[
                    cluster] * abs(
                    x[i] ** shape[cluster]) / (
                                              x[i])

                test_mk_den[i, cluster] = fik[i][cluster] * rnk[i][cluster] * e_precision_[cluster] * abs(x[i]) ** \
                                          shape[cluster] * shape[
                                              cluster] * (
                                                  shape[cluster] - 1) / (2 * x[i] ** 2)
            else:
                test_mk_den[i, cluster] = fik[i][cluster] * rnk[i][cluster] * e_precision_[cluster] * mk[cluster] ** (
                        shape[cluster] - 2)

                if test_mk_den[i, cluster] <= 0:
                    test_mk_den[i, cluster] = 0

                term0 = fik[i][cluster] * rnk[i][cluster] * e_precision_[cluster] * shape[cluster] / 2 * mk[
                    cluster] ** (
                                shape[cluster] - 2) * x[i]

                term1 = fik[i][cluster] * rnk[i][cluster] * e_precision_[cluster] * shape[cluster] / 4 * (
                        shape[cluster] - 1) * mk[
                            cluster] ** (shape[cluster] - 3) * x[i] ** 2

                if term1 <= 0:
                    term1 = 0

                if term0 <= 0:
                    term0 = 0

                test_mk_num[i, cluster] = term1 + term0

        numerator = test_mk_num.sum(axis=0)[cluster] + s0[cluster] * m0[cluster] / 2
        sk[cluster] = test_mk_den.sum(axis=0)[cluster] + s0[cluster] / 2

        mk[cluster] = numerator / sk[cluster]
    print(mk, " Shape : ", e_precision_)
    print(f_mean, " Shape : ", f_precision)
    # print()
    gammak = gama0 + Nk
    e_ln_pi = e_ln_pi_k(gammak, Nk)

    e_ln_precision_ = digamma(alphak) - np.log(np.ma.array(betak))

    e_precision_ = alphak / betak

    for cluster in range(k):
        for i in range(len(x)):
            if x[i] > mk[cluster]:
                if x[i] != 0:
                    e_x_mean_lambda_[i, cluster] = x[i] ** shape[cluster] - \
                                                   shape[cluster] * x[i] ** shape[cluster] / x[i] * mk[cluster] + \
                                                   shape[cluster] / 2 * (shape[cluster] - 1) * x[i] ** shape[cluster] / \
                                                   x[i] ** 2 * (
                                                           1 / sk[cluster] +
                                                           mk[cluster] ** 2)

            else:

                t1 = -shape[cluster] * x[i] * e_mean_n(sk, mk, shape[cluster] - 1, k)
                e_mean2 = e_mean_n(sk, mk, shape[cluster] - 2, k)

                t2 = [0, 0]
                if shape[cluster] > 1:
                    t2 = shape[cluster] / 2 * (shape[cluster] - 1) * x[i] ** 2 * e_mean2

                e_x_mean_lambda_[i, cluster] = e_mean_n(sk, mk, shape[cluster], k)[cluster] + t1[cluster] + t2[cluster]

            e_x_mean_lambda_[i, cluster] = abs(e_x_mean_lambda_[i, cluster])
    count += 1
    # L1 = lowerbound_first_dir(rnk, x, shape, mk, s0, e_precision_).sum(axis=0)
    # L2 = lowerbound_second_dir(rnk, x, shape, mk, s0, e_precision_).sum(axis=0)
    # L1 += sys.float_info.epsilon
    # L2 += sys.float_info.epsilon
    # delta_lowerbound = L1 / L2
    # delta_lowerbound += sys.float_info.epsilon
    # shape = shape - 0.01 * (delta_lowerbound)

    result = []
    counter = 0
    segmentedImage = np.zeros((len(x), 1), np.uint8)

    # assigning values to pixels of different segments
    rnk1 = list(rnk)
    rnk1 = [list(i) for i in rnk1]
    m12 = [0, 100]
    for response in rnk1:
        maxResp = max(response)
        respmax = response.index(maxResp)
        result.append(respmax)
        segmentedImage[counter] = m0[respmax]
        counter = counter + 1

    segmentedImage = segmentedImage.reshape(o_shape[0], o_shape[1])

    # pyplot.subplot(3, 1, 3)

    pyplot.imshow(segmentedImage)
    # pyplot.savefig("/Users/Srikanth/PycharmProjects/DataMiningClass/outputs/starfish/" + str(count) + ".png")
    pyplot.show()
