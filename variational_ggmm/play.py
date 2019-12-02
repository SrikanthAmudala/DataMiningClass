import cv2

from scipy.special import gamma
from scipy.special import digamma
import math
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from matplotlib import pyplot


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


def e_x_mean_lamnda_2_multid(mk, x, s0):
    # x^2
    y = [np.dot(i, i.T) for i in x]
    # m^2
    mk_sq = [np.dot(i, i.T) for i in mk]

    temp1 = []
    for j in mk:
        temp = []
        for i in x:
            temp.append(np.dot(i.reshape(-1, 1).T, j.reshape(-1, 1)))
        temp1.append(temp)
    temp1 = np.asarray(temp1).reshape(-1, 1, 2)
    # -2 x m
    temp1 = -2 * temp1
    # x^2 - 2 x m
    temp = [i - j for i, j in zip(temp1, y)]

    # temp + m^2
    final = [j + mk_sq for j in temp]
    final = np.asarray(final)
    return 1 / s0 + final
    # # y = (2 * mk.reshape(-1, 1) * x)
    # y1 = np.power(x, 2) + np.power(mk, 2).reshape(-1, 1)
    # return (1 / s0).reshape(-1, 1) + y1 - y


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
    return digamma(alpha) - np.log(np.abs(beta))


# E[T]
def e_precision(alpha, beta):
    return alpha / beta


def e_precision_2(alphak, betak, shape):
    y = 1 / (gamma(alphak) * np.power(betak, 2 - shape)) * gamma(2 + alphak + shape)
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
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x)

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


input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample_copy.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample.jpg"
# input data points
x = cv2.imread(input_img_path)

pyplot.subplot(3, 1, 1)
pyplot.imshow(x)
o_shape = x.shape
dimention = o_shape[-1]

x = x.reshape(-1, dimention)

k = 2
# hyper parameters


alpha0 = np.asarray([0.1 for _ in range(k)])
beta0 = np.asarray([1 / len(x) for _ in range(k)])
shape = np.asarray([2 for _ in range(k)])
m0, s0, gama0, resp = initilization(k, x)

pyplot.subplot(3, 1, 2)
pyplot.imshow(resp.reshape(o_shape[0], o_shape[1]))
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

z_list = []

# for Multi Dimension
# mk = m0 + (Nk * shape * e_precision_2(alpha0, beta0, shape)).reshape(-1,1)
mk = m0[:, i] + Nk * shape * e_precision_2(alpha0, beta0, shape)
# e_x_mean_lambda_ = e_x_mean_lambda_short(s0, x, shape, mk, m0)

print("mk: ", mk)
# e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x[:, i], s0).T
e_x_mean_lambda_ = e_x_mean_lamnda_2_multid(mk, x, s0)

count = 0
# while
while (count < 10):
    z = e_ln_pi + (1 / shape) * e_ln_precision_ + np.log(shape) - np.log(
        gamma(1 / shape)) - e_precision_ * e_x_mean_lambda_
    print("Z: ", z[0])
    rnk = z / z.sum(axis=0)
    Nk = rnk.sum(axis=0)
    alphak = Nk / shape + alpha0 - 1
    betak = beta0 + Nk * e_x_mean_lambda_.sum(axis=0)
    e_precision_2_k = e_precision_2(alphak, betak, shape)
    e_ln_pi = e_ln_pi_k(gama0, Nk)
    e_ln_precision_ = e_ln_precision(alphak, betak)
    e_precision_ = e_precision(alphak, betak)

    mk = m0 + (Nk * shape * e_precision_2(alpha0, beta0, shape)).reshape(-1, 1)
    # mk = m0.reshape(-1) + Nk * shape * e_precision_2_k
    # e_x_mean_lambda_ = e_x_mean_lambda_short(s0, x, shape, mk, m0)
    # e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, s0).T
    e_x_mean_lambda_ = e_x_mean_lamnda_2_multid(mk, x, s0)
    count += 1


"""


alphak = Nk / shape + alpha0 - 1
betak = beta0 + Nk * e_x_mean_lambda_.sum(axis=0)
e_precision_2_k = e_precision_2(alphak, betak, shape)
mk = m0 + (Nk * shape * e_precision_2_k).reshape(-1, 1)

e_ln_precision_ = e_ln_precision(alpha0, beta0)
e_precision_ = e_precision(alpha0, beta0)

# e_x_mean_lambda_ = e_x_mean_lambda(s, means, x, shape, m)


ln_row_nk_ = ln_row_nk(e_ln_pi, e_ln_precision_, e_precision_, e_x_mean_lambda_, shape)
rnk = ln_row_nk_ / ln_row_nk_.sum()
Nk = rnk.sum(axis=0)

# m = np.array([np.ones(k) for _ in range(len(x))])


# While loop
count = 0
while count < 50:
    print(count)
    alpha = Nk / shape + alpha - 1

    # should not give -ve values, its -ve as e_x_mean_lambda ~ -ve
    beta = beta + (Nk * e_x_mean_lambda_.sum(axis=0))

    sign_decider = (1 - (x - means).T * precision) * shape
    m = m + Nk * shape * e_precision(alpha, beta) / s

    # check your means its becoming -ve
    means = -(s * (means.T - m) / 2).T
    # e_x_mean_lambda_ = e_x_mean_lambda(s, means, x, shape, m)
    e_x_mean_lambda_ = e_x_mean_lambda_short(s, means, x, shape, m)

    #
    # for i in range(k):
    #     for j in range(len(m[:, i])):
    #         if sign_decider[j, k] < 1:
    #             m[j, i] = m[j, i] - m1[j, i]
    #         else:
    #             m[j, i] = m[j, i] + m1[j, i]

    gama = gama + Nk
    e_ln_pi = e_ln_pi_k(gama)
    e_precision_ = e_precision(alpha, beta)
    e_ln_precision_ = e_ln_precision(alpha, beta)
    ln_row_nk_ = ln_row_nk(e_ln_pi, e_ln_precision_, e_precision_, e_x_mean_lambda_, shape)
    rnk = ln_row_nk_ / ln_row_nk_.sum()
    Nk = rnk.sum(axis=0)
    count += 1
"""

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
    segmentedImage[counter] = 255 - mk[respmax]
    counter = counter + 1

segmentedImage = segmentedImage.reshape(o_shape[0], o_shape[1])

pyplot.subplot(3, 1, 3)
pyplot.imshow(segmentedImage)
pyplot.show()
