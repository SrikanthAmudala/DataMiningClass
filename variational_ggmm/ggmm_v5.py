"""
Variational GGMM with Multi dimention Img
"""

import cv2
import sys
from scipy.special import gamma
from scipy.special import digamma
from scipy import special
import math
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from matplotlib import pyplot


# E[ln pik]
def e_ln_pi_k(gama0, Nk):
    gammak = gama0 + Nk
    return digamma(gammak) - digamma(gammak.sum())


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


def e_x_mean_lamnda_1(mk, x):
    return x.reshape(-1, 1) - mk


# E[|x-u|^lambda]
def e_x_mean_lambda(s, means, x, shape, m):
    e_mean = (shape * (means - x).T) * (np.log(np.abs((means - x).T / np.sqrt(s))) - 1) + (
        np.log(np.sqrt(s / 2 * math.pi))).reshape(1, -1) - s * (np.power(means - m.reshape(-1, 1), 3)).reshape(1,
                                                                                                               -1) / 6
    return e_mean
    # return np.log(np.sqrt(s / 2 * math.pi))


def e_x_mean_lambda_short(s, x, shape, m, m0):
    y = - (m.reshape(-1, 1) * s.reshape(-1, 1) / 3) * (
            np.power(m, 2).reshape(-1, 1) + 3 * np.power(m0.reshape(-1, 1), 2))

    y1 = (shape * m.T).reshape(-1, 1) * np.log(np.abs(np.power(x, 2) - np.power(m, 2).reshape(-1, 1)))

    y2 = (np.log(np.sqrt(s / 2 * math.pi)).reshape(-1, 1))

    y3 = y + y1 + y2
    return y3.T


def e_x_mean_lamnda_2(mk, x, s0):
    y = (2 * mk.reshape(-1, 1) * x)
    y1 = np.power(x, 2) + np.power(mk, 2).reshape(-1, 1)
    return (1 / s0).reshape(-1, 1) + y1 - y


def exp_mean_lambda_n(s, means, x, shape, m):
    test = (shape * (means - x).T) * (np.log(np.abs((means - x).T / np.sqrt(s))) - 1) + (
        np.log(np.sqrt(s / 2 * math.pi))).reshape(1, -1) - s * (np.power(means.T - m, 3))
    return test


# E[ln(T)]
def e_ln_precision(alpha, beta):
    return digamma(alpha) - np.log(np.abs(beta))


# E[T]
def e_precision(alpha, beta):
    return alpha / beta


#
# def e_precision_2(alphak, betak, shape):
#     y = 1 / (gamma(alphak) * np.power(betak, 2 - shape)) * gamma(2 + alphak + shape)
#     return y


def e_precision_2(alphak, betak, shape):
    # y = 1 / (gamma(alphak) * np.power(betak, 2 - shape)) * gamma(2 + alphak - shape)
    p1 = 2 - shape
    p1 = [int(i) for i in p1]
    y = 1 / (gamma(alphak) * betak ** p1) * gamma(2 + alphak - shape)
    return y


def p_z_pi():
    pass


def p_pi():
    pass


def p_mean():
    pass


def p_precision():
    pass


def q_z():
    pass


def q_pi():
    pass


def q_mean():
    pass


def q_precision():
    pass


def ln_row_nk(e_ln_pi, e_ln_precision_, e_precision_, e_x_mean_lambda_, shape):
    ln_row_nk_ = e_ln_pi + (1 / shape) * e_ln_precision_ + np.log(shape) - np.log(
        gamma(1 / shape)) - e_precision_ * e_x_mean_lambda_
    return ln_row_nk_


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
    print("var: ", variance)
    precision = 1 / variance
    weights = []

    for i in clustered_img.keys():
        weights.append(len(clustered_img.get(i)) / len(x))

    weights = np.asarray(weights)

    # resp = np.hstack((labels.reshape(-1,1), labels_1.reshape(-1, 1)))
    return (means, precision, weights, labels)


# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample_copy.jpg"
input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/demo.png"
# input data points

img = cv2.imread(input_img_path)
k = 3
d = 3
pyplot.subplot(3, 1, 1)
pyplot.imshow(img)
temp_rnk = []
for t1 in range(d):
    x = img[:, :, t1]
    o_shape = x.s
    x = x.reshape(-1)
    # hyper parameters

    alpha0 = np.asarray([0.1 for _ in range(k)])
    beta0 = np.asarray([1 / len(x) for _ in range(k)])
    shape = np.asarray([2 for _ in range(k)])
    m0, s0, gama0, resp = initilization(k, x)

    temp = np.zeros((len(x), k))
    z = []

    for i, j in zip(resp, temp):
        j[i] = 1
        z.append(j)

    z = np.asarray(z)
    rnk = z / z.sum(axis=0)
    Nk = rnk.sum(axis=0)
    e_ln_pi = e_ln_pi_k(gama0, Nk)
    e_ln_precision_ = e_ln_precision(alpha0, beta0)
    e_precision_ = e_precision(alpha0, beta0)
    mk = m0.reshape(-1) + Nk * shape * e_precision_2(alpha0, beta0, shape) / s0
    # e_x_mean_lambda_ = e_x_mean_lambda_short(s0, x, shape, mk, m0)

    # e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, s0).T

    if shape.max() > 1.5:
        e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, s0).T
    else:
        e_x_mean_lambda_ = e_x_mean_lamnda_1(mk, x)

    count = 0
    while (count < 50):

        z = e_ln_pi + (1 / shape) * e_ln_precision_ + np.log(shape) - np.log(
            gamma(1 / shape)) - e_precision_ * e_x_mean_lambda_
        # rnk = z / z.sum(axis=0)
        rnk = z / z.sum(axis=1).reshape(-1, 1)
        Nk = rnk.sum(axis=0)
        L1 = lowerbound_first_dir(rnk, x, shape, mk, s0, e_precision_).sum(axis=0)
        L2 = lowerbound_second_dir(rnk, x, shape, mk, s0, e_precision_).sum(axis=0)
        L1 += sys.float_info.epsilon
        L2 += sys.float_info.epsilon
        delta_lowerbound = L1 / L2
        delta_lowerbound += sys.float_info.epsilon
        shape = shape - 0.01 * (delta_lowerbound)
        print(t1, ": Shape: ", shape)
        alphak = Nk / shape + alpha0 - 1
        betak = beta0 + Nk * e_x_mean_lambda_.sum(axis=0)
        e_precision_2_k = e_precision_2(alphak, betak, shape)

        e_ln_pi = e_ln_pi_k(gama0, Nk)
        e_ln_precision_ = e_ln_precision(alphak, betak)
        e_precision_ = e_precision(alphak, betak)
        # print("Mk: ", mk)
        mk = m0.reshape(-1) + Nk * shape * e_precision_2_k
        # e_x_mean_lambda_ = e_x_mean_lambda_short(s0, x, shape, mk, m0)
        # e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, s0).T

        if shape.max() > 1.5:
            e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, s0).T
        else:
            e_x_mean_lambda_ = e_x_mean_lamnda_1(mk, x)

        count += 1
    temp_rnk.append(rnk)

rnk_prod = 0
for i in range(d):
    rnk_prod += temp_rnk[i]

result = []
counter = 0
segmentedImage = np.zeros((len(x), 1), np.uint8)

# assigning values to pixels of different segments
rnk = list(rnk_prod)
rnk = [list(i) for i in rnk]
for response in rnk:
    maxResp = max(response)
    respmax = response.index(maxResp)
    result.append(respmax)
    segmentedImage[counter] = 255 - mk[respmax]
    counter = counter + 1

segmentedImage = segmentedImage.reshape(o_shape[0], o_shape[1])

pyplot.subplot(3, 1, 3)
pyplot.imshow(segmentedImage)
pyplot.show()
