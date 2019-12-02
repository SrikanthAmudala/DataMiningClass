"""
Variational GGMM with Multi dimention Irish and well defined shape
"""
import cv2

from scipy.special import gamma
from scipy.special import digamma
import math
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from matplotlib import pyplot
from sklearn.datasets import load_iris


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


# E[ln(T)]
def e_ln_precision(alpha, beta):
    return digamma(alpha) - np.log(np.abs(beta))


# E[T]
def e_precision(alpha, beta):
    return alpha / beta


def e_precision_2(alphak, betak, shape):
    y = 1 / (gamma(alphak) * np.power(betak, 2 - shape)) * gamma(2 + alphak + shape)
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


from numpy import *


def gen(K, N, XDim):
    # K: number of components
    # N: number of data points

    mu = np.array([np.random.multivariate_normal(zeros(XDim), 10 * eye(XDim)) for _ in range(K)])
    cov = [0.1 * eye(XDim) for _ in range(K)]
    q = random.dirichlet(ones(K))  # component coefficients
    X = zeros((N, XDim))  # observations
    Z = zeros((N, K))  # latent variables
    for n in range(N):
        # decide which component has responsibility for this data point:
        Z[n, :] = random.multinomial(1, q)
        k = Z[n, :].argmax()
        X[n, :] = random.multivariate_normal(mu[k, :], cov[k])
    return X


# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample_copy.jpg"
input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample.jpg"
# input data points
iris = load_iris()
# X = iris.data[:, :4]  # we only take the first two features.

X = gen(30, 200, 2)
y = iris.target
# img = cv2.imread(input_img_path)
k = 20
d = 2

temp_rnk = []
for t1 in range(d):
    x = X[:, t1]
    o_shape = x.shape
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

    e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, s0).T
    count = 0
    # while
    while (count < 10):
        z = e_ln_pi + (1 / shape) * e_ln_precision_ + np.log(shape) - np.log(
            gamma(1 / shape)) - e_precision_ * e_x_mean_lambda_
        # print("Z: ", z[0])
        rnk = z / z.sum(axis=0)
        Nk = rnk.sum(axis=0)

        alphak = Nk / shape + alpha0 - 1
        betak = beta0 + Nk * e_x_mean_lambda_.sum(axis=0)
        e_precision_2_k = e_precision_2(alphak, betak, shape)

        e_ln_pi = e_ln_pi_k(gama0, Nk)
        e_ln_precision_ = e_ln_precision(alphak, betak)
        e_precision_ = e_precision(alphak, betak)
        print("Mk: ", mk)
        mk = m0.reshape(-1) + Nk * shape * e_precision_2_k
        # e_x_mean_lambda_ = e_x_mean_lambda_short(s0, x, shape, mk, m0)
        e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, s0).T
        count += 1

    temp_rnk.append(rnk)
rnk_prod = 0
for i in range(d):
    rnk_prod += temp_rnk[i]

result = []

# assigning values to pixels of different segments
rnk = list(rnk_prod)
rnk = [list(i) for i in rnk]
for response in rnk:
    maxResp = max(response)
    respmax = response.index(maxResp)
    result.append(respmax)


# unique_true, counts_true = np.unique(y, return_counts=True)
# result = np.asarray(result)
# unique_tar, counts_tar = np.unique(result, return_counts=True)
#

def naming(labels):
    t1 = list(labels)
    for n, i in enumerate(t1):
        if i == 2:
            t1[n] = 'two'
        elif i == 0:
            t1[n] = 'one'
        elif i == 1:
            t1[n] = 'zero'

    for n, i in enumerate(t1):
        if i == 'zero':
            t1[n] = 0
        elif i == 'one':
            t1[n] = 1
        elif i == 'two':
            t1[n] = 2
