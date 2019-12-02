"""
GGMM with well defined shape parameter
"""

from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot
from scipy.special import digamma
from scipy.special import gamma
from scipy import special
from sklearn.cluster import KMeans


# E[ln pik]
def e_ln_pi_k(gama0, Nk):
    gammak = gama0 + Nk
    return digamma(gammak) - digamma(gammak.sum())


def e_x_mean_lamnda_2(mk, x, s0):

    return 1/s0.reshape(-1,1) + (x.reshape(-1,1) - mk.reshape(-1)).T
    y = (2 * mk.reshape(-1, 1) * x)
    y1 = np.power(x, 2) + np.power(mk, 2).reshape(-1, 1)
    return (1 / s0).reshape(-1, 1) + y1 - y


def e_x_mean_lamnda_1(mk, x):
    return x.reshape(-1, 1) - mk

    # y = (2 * mk.reshape(-1, 1) * x)
    # y1 = np.power(x, 2) + np.power(mk, 2).reshape(-1, 1)
    # return (1 / s0).reshape(-1, 1) + y1 - y


def e_x_mean_lamnda_3(mk, x, s0):
    y = (2 * mk.reshape(-1, 1) * x)
    y1 = np.power(x, 2) + np.power(mk, 2).reshape(-1, 1)
    return (1 / s0).reshape(-1, 1) + y1 - y


# E[ln(T)]
def e_ln_precision(alpha, beta):
    return digamma(alpha) - np.log(np.abs(beta))


# E[T]
def e_precision(alpha, beta):
    return alpha / beta


def e_precision_2(alphak, betak, shape):
    # y = 1 / (gamma(alphak) * np.power(betak, 2 - shape)) * gamma(2 + alphak - shape)
    p1 = 2 - shape
    p1 = [int(i) for i in p1]
    y = 1 / (gamma(alphak) * betak ** p1) * gamma(2 + alphak - shape)
    return y


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
    precision = 1 / variance
    weights = []

    for i in clustered_img.keys():
        weights.append(len(clustered_img.get(i)) / len(x))

    weights = np.asarray(weights)

    # resp = np.hstack((labels.reshape(-1,1), labels_1.reshape(-1, 1)))
    return (means, precision, weights, labels)


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


# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample_copy.jpg"
input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample.jpg"
# input data points
x = cv2.imread(input_img_path, 0)

pyplot.subplot(3, 1, 1)
pyplot.imshow(x)
o_shape = x.shape
x = x.reshape(-1)
k = 2

# hyper parameters

alpha0 = np.asarray([0.1 for _ in range(k)])
beta0 = np.asarray([1 / len(x) for _ in range(k)])
shape = np.asarray([2 for _ in range(k)])
m0, s0, gama0, resp = initilization(k, x)

pyplot.subplot(3, 1, 2)
pyplot.imshow(resp.reshape(o_shape))
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

precision = 0  # calculate precision

# e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, s0).T

# if shape.max()>1.5:
#     e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, s0).T
# else:
#     e_x_mean_lambda_ = e_x_mean_lamnda_1(mk, x)
#
e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, s0).T
count = 0

while (count < 1000):
    z = e_ln_pi + (1 / shape) * e_ln_precision_ + np.log(shape) - np.log(
        gamma(1 / shape)) - e_precision_ * e_x_mean_lambda_
    # print("Z: ", z[0])
    print(mk)
    rnk = z / z.sum(axis=1).reshape(-1,1)
    Nk = rnk.sum(axis=0)
    L1 = lowerbound_first_dir(rnk, x, shape, mk, s0, e_precision_).sum(axis=0)
    L2 = lowerbound_second_dir(rnk, x, shape, mk, s0, e_precision_).sum(axis=0)
    delta_lowerbound = L1 / L2
    shape = shape - 0.01 * (delta_lowerbound)
    alphak = Nk / shape + alpha0 - 1
    betak = beta0 + Nk * e_x_mean_lambda_.sum(axis=0)
    e_precision_2_k = e_precision_2(alphak, betak, shape)
    e_ln_pi = e_ln_pi_k(gama0, Nk)
    e_ln_precision_ = e_ln_precision(alphak, betak)
    e_precision_ = e_precision(alphak, betak)
    # print("Mk: ", mk)
    # mk = m0.reshape(-1) + Nk * shape * e_precision_2_k


    # e_x_mean_lambda_ = e_x_mean_lambda_short(s0, x, shape, mk, m0)
    # e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, s0).T

    mk_sign = shape.reshape(-1, 1) * (1 - (x - m0) * e_precision_.reshape(-1, 1))

    mk_sign = mk_sign.sum(axis=1)
    mk = []

    for i in range(0, k):
        if mk_sign[i] > 0:
            mk_temp = m0[i] + Nk[i] * shape[i] * e_precision_2_k[i] / s0[i]
        else:
            mk_temp = m0[i] - Nk[i] * shape[i] * e_precision_2_k[i] / s0[i]
        mk.append(mk_temp)
    mk = np.asarray(mk).reshape(-1)
    if shape.max() > 1.5:
        e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, s0).T
    else:
        e_x_mean_lambda_ = e_x_mean_lamnda_1(mk, x)

    count += 1

result = []
counter = 0
segmentedImage = np.zeros((len(x), 1), np.uint8)

# assigning values to pixels of different segments
rnk = list(rnk)
rnk = [list(i) for i in rnk]
for response in rnk:
    maxResp = max(response)
    respmax = response.index(maxResp)
    result.append(respmax)
    segmentedImage[counter] = 255 - m0[respmax]
    counter = counter + 1

segmentedImage = segmentedImage.reshape(o_shape[0], o_shape[1])

pyplot.subplot(3, 1, 3)
pyplot.imshow(segmentedImage)
pyplot.show()
