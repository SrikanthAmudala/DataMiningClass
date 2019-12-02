"""
Fixing the gmm with rnk
"""

import math
from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot
from scipy.special import digamma
from scipy.special import gamma
from sklearn.cluster import KMeans


def alpha_n():
    pass


def beta_n():
    pass


def s_n():
    pass


def m_n():
    pass


# E[ln pik]
def e_ln_pi_k(gama0, Nk):
    gammak = gama0 + Nk
    return digamma(gammak) - digamma(gammak.sum())


# E[T^2-lambda]
def e_t_two_lambda():
    pass


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


# E[mean]
def e_mean():
    pass


# E[mean^2]
def e_mean_two():
    pass


# E[ln(T)]
def e_ln_precision(alpha, beta):
    # return digamma(alpha) - np.log(np.abs(beta))
    # print("alpha: ", alpha)
    # print("diagamma: ", digamma(alpha))
    # print("beat: ", np.log(beta))
    # print("beat##### ",beta)
    return digamma(alpha) - np.log(beta)


# E[T]
def e_precision(alpha, beta):
    return alpha / beta


def e_precision_2(alphak, betak, shape):
    # return alphak / betak
    # p1 = gamma(alphak) * np.power(betak, 2 - shape)
    # temp = []
    # for i in p1:
    #     if i==np.inf:
    #         temp.append(0)
    #     else:
    #         temp.append(1/i)
    #
    # p2 = gamma(2 + alphak + shape)
    # # temp1 = []
    # # for i in p2:
    # #     if i==np.inf:
    # #         temp1.append(10000000000)
    # #     else:
    # #         temp1.append(i)
    # y1 = np.asarray(temp)
    # # y2 = np.asarray(temp1)
    # return y1*p2
    #

    y = 1 / (gamma(alphak) * np.power(betak, 2 - shape)) * gamma(2 + alphak - shape)
    # y = 1 / (digamma(alphak) * np.power(betak, 2 - shape)) * digamma(2 + alphak + shape)

    #
    # # y = 1 / (scipy.special.gammaln(alphak) * np.power(betak, 2 - shape)) * scipy.special.gammaln(2 + alphak + shape)
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
    precision = 1 / variance
    weights = []

    for i in clustered_img.keys():
        weights.append(len(clustered_img.get(i)) / len(x))

    weights = np.asarray(weights)

    # resp = np.hstack((labels.reshape(-1,1), labels_1.reshape(-1, 1)))
    return (means, precision, weights, labels)


# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample_copy.jpg"
input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample.jpg"
# input data points
x = cv2.imread(input_img_path, 0)
N = len(x)
pyplot.subplot(3, 1, 1)
pyplot.imshow(x)
o_shape = x.shape
x = x.reshape(-1)
k = 2

# hyper parameters
alpha0 = np.asarray([x.mean() ** 2 / x.var() ** 2 for _ in range(k)])
# beta0 = np.asarray([1 / len(x) for _ in range(k)])
beta0 = np.asarray([x.var() / x.mean() for _ in range(k)])

shape = np.asarray([2 for _ in range(k)])
m0, s0, gama0, resp = initilization(k, x)

# m0 = np.asarray([0, 100, 200]).reshape(-1, 1)
# s0 = np.asarray([0.01,0.01,0.01]).reshape(-1)
pyplot.subplot(3, 1, 2)
pyplot.imshow(resp.reshape(o_shape))
temp = np.zeros((len(x), k))
z = []

for i, j in zip(resp, temp):
    j[i] = 1
    z.append(j)

z = np.asarray(z)
# rnk = z / z.sum(axis=0)
# rnk = z / z.sum(axis=1).reshape(-1, 1)
rnk = np.exp(z) / np.reshape(np.exp(z).sum(axis=1), (-1, 1))
Nk = rnk.sum(axis=0)

e_ln_pi = e_ln_pi_k(gama0, Nk)
e_ln_precision_ = e_ln_precision(alpha0, beta0)
e_precision_ = e_precision(alpha0, beta0)

# mk_sign = shape.reshape(-1, 1) * (1 - (x - m0) * e_precision_.reshape(-1, 1))
#
#
# mk_sign = mk_sign.sum(axis=1)
# mk = []
e_precision_2_k = e_precision_2(alpha0, beta0, shape)
#
# for i in range(0, k):
#     if mk_sign[i] > 0:
#         mk_temp = m0[i] + Nk[i] * shape[i] * e_precision_2_k[i] / s0[i]
#     else:
#         mk_temp = m0[i] - Nk[i] * shape[i] * e_precision_2_k[i] / s0[i]
#     mk.append(mk_temp)
# mk = np.asarray(mk).reshape(-1)
# # e_x_mean_lambda_ = e_x_mean_lambda_short(s0, x, shape, mk, m0)

mk = m0
sk = s0 - Nk*e_precision_

# mk = m0.reshape(-1) + Nk * shape * e_precision_2_k / s0
e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, s0).T

count = 0
# while

while (count < 100):
    for i in range(k):
        z[:, i] = e_ln_pi[i] + (1 / shape[i]) * e_ln_precision_[i] + np.log(shape[i]) - \
                  np.log(2 * gamma(1 / shape[i])) - e_precision_[i] * e_x_mean_lambda_[:, i]
        # z[:, i] = abs(e_precision_[i] * e_x_mean_lambda_[:, i])

    # rnk = np.exp(z) / np.reshape(np.exp(z).sum(axis=1), (-1, 1))
    # print("Z: ", z[0])
    # rnk = z / z.sum(axis=0)
    rnk = z / z.sum(axis=1).reshape(-1, 1)
    print("rnk: ", rnk[0])
    # Nk = rnk.sum(axis=0)
    NK = rnk.sum(axis=0)
    print(Nk)


    alphak = Nk / shape + alpha0 - 1
    # betak = beta0 + np.abs(Nk * e_x_mean_lambda_.sum(axis=0))
    betak = beta0 + (rnk * e_x_mean_lambda_).sum(axis=0)

    print(alphak, betak)
    # betak = beta0 + Nk
    # betak = beta0 + np.abs((rnk * e_x_mean_lambda_).sum(axis=0))
    e_precision_2_k = e_precision_2(alphak, betak, shape)

    # print(e_precision_2_k)
    e_ln_pi = e_ln_pi_k(gama0, Nk)
    e_ln_precision_ = e_ln_precision(alphak, betak)
    e_precision_ = e_precision(alphak, betak)

    # mk_sign = shape.reshape(-1, 1) * (1 - (x - m0) * e_precision_.reshape(-1, 1))
    # mk_sign = mk_sign.sum(axis=1)
    # mk = []
    #
    # for i in range(0, k):
    #     if mk_sign[i] > 0:
    #         mk_temp = m0[i] + Nk[i] * shape[i] * e_precision_2_k[i] / s0[i]
    #     else:
    #         mk_temp = m0[i] - Nk[i] * shape[i] * e_precision_2_k[i] / s0[i]
    #     mk.append(mk_temp)
    # mk = np.asarray(mk).reshape(-1)

    # mk = m0.reshape(-1) + Nk * shape * e_precision_2_k / s0
    print("Mk: ", mk)
    # e_x_mean_lambda_ = e_x_mean_lambda_short(s0, x, shape, mk, m0)
    mk = m0
    sk = s0 - Nk * e_precision_
    e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, sk).T
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
