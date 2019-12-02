"""
shape = 2, binomial expansion
"""

import math
from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot
from scipy.special import digamma
from scipy.special import gamma
from sklearn.cluster import KMeans

import mpmath


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
    for cluster in range(k):
        p = (1 / math.sqrt(sk[cluster])) ** shape * 2 ** (shape / 2) * gamma((1 + shape) / 2) / math.sqrt(
            math.pi) * mpmath.hyp1f1(-shape / 2, 1 / 2, -1 / 2 * mk[cluster] ** 2 * sk[0])
        temp.append(p)
    return np.asarray(temp).reshape(-1, 1)


# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/starfish.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testimg.png"
input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample_copy.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/circle.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/birds2.jpg"
# input_img_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/demo.png"

x = cv2.imread(input_img_path, 0)
N = len(x)
o_shape = x.shape
x = x.reshape(-1)
x = x + 0.0
k = 2
alpha0 = np.asarray([x.mean() ** 2 / x.var() for _ in range(k)])
beta0 = np.asarray([x.mean() / i for i in alpha0])
# beta0 = np.asarray([(1e-20) * 1 / i for i in alpha0])
shape = 1.4

m0, s0, gama0, resp = initilization(k, x)

# s0 = 1/s0

# pyplot.subplot(3, 1, 2)
pyplot.imshow(resp.reshape(o_shape))
pyplot.show()
temp = np.zeros((len(x), k))
z = []

for i, j in zip(resp, temp):
    j[i] = 1
    z.append(j)

# z = np.asarray(z)
z = np.array([np.random.dirichlet(np.ones(k)) for _ in range(len(x))])
rnk = np.exp(z) / np.reshape(np.exp(z).sum(axis=1), (-1, 1))
Nk = rnk.sum(axis=0)
sk = s0 + 2 * Nk * (alpha0 / beta0)

test_mk_num = z.copy()
test_mk_den = z.copy()

mk = m0.copy()

e_precision_ = alpha0 / beta0

"""
for cluster in range(k):
    for i in range(len(x)):
        if x[i] > m0[cluster]:
            test_mk_num[i, cluster] = rnk[i][cluster] * e_precision_[cluster]*shape*abs(x[i]**shape)/(2*x[i])
            test_mk_den[i, cluster] = rnk[i][cluster]*e_precision_[cluster]*abs(x[i])**shape * shape*(shape-1)/(2*x[i]**2)
        
        
        else:

            test_mk_den[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * mk[cluster]**(shape-2)
            test_mk_num[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * shape/2 * mk[cluster] ** (shape-2) * x[i] - \
                          rnk[i][cluster] * e_precision_[cluster] * shape/4 * (shape-1) * mk[cluster]**(shape-3) * x[i]**2


            # test_mk_num[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * (255 ** (shape - 1)) * (
            #             shape - (1 - x[i]) * shape * (shape - 1))
            # test_mk_den[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * (255 ** (shape - 1)) * shape * (
            #             shape - 1)
"""

for cluster in range(k):
    for i in range(len(x)):
        if x[i] > mk[cluster]:
            test_mk_num[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * shape * abs(x[i] ** shape) / (
                    2 * x[i])

            test_mk_den[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * abs(x[i]) ** shape * shape * (
                    shape - 1) / (2 * x[i] ** 2)
        else:

            test_mk_den[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * mk[cluster] ** (shape - 2)
            if test_mk_den[i, cluster] <= 0:
                test_mk_den[i, cluster] = 0

            # term0 = 0
            # term1 = 0
            # if shape >= 2:
            term0 = rnk[i][cluster] * e_precision_[cluster] * shape / 2 * mk[cluster] ** (
                    shape - 2) * x[i]

            term1 = rnk[i][cluster] * e_precision_[cluster] * shape / 4 * (shape - 1) * mk[
                cluster] ** (shape - 3) * x[i] ** 2

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
                e_x_mean_lambda_[i, cluster] = x[i] ** shape - shape * x[i] ** shape / x[i] * mk[
                    cluster] + shape / 2 * (
                                                       shape - 1) * x[i] ** shape / x[i] ** 2 * (
                                                       1 / sk[cluster] + mk[cluster] ** 2)

        else:

            t1 = -shape * x[i] * e_mean_n(sk, mk, shape - 1, k)
            e_mean2 = e_mean_n(sk, mk, shape - 2, k)
            t2 = [0, 0]

            if shape > 1:
                t2 = shape / 2 * (shape - 1) * x[i] ** 2 * e_mean2

            e_x_mean_lambda_[i, cluster] = e_mean_n(sk, mk, shape, k)[cluster] + t1[cluster]
            e_x_mean_lambda_[i, cluster] = e_x_mean_lambda_[i, cluster] + t2[cluster]
            # e_x_mean_lambda_[i, cluster] = e_mean_n(sk, mk, shape, k)[cluster] - \
            #                                shape * x[i] * e_mean_n(sk, mk, shape - 1, k)[cluster] + \
            #                                shape / 2 * (shape - 1) * x[i] ** 2 * e_mean_n(sk, mk, shape - 2, k)[cluster]
            e_x_mean_lambda_[i, cluster] = abs(e_x_mean_lambda_[i, cluster])
            # if e_x_mean_lambda_[i, cluster] < 0:
            #     e_x_mean_lambda_[i, cluster] = 0
count = 0

while (count < 50):
    for i in range(k):
        p1 = e_ln_pi[i] + (1 / shape) * e_ln_precision_[i] + np.log(shape) - np.log(2 * gamma(1 / shape))
        z[:, i] = (p1 - (e_precision_[i] * e_x_mean_lambda_[:, i]).reshape(-1, 1)).reshape(-1)

    rnk = np.exp(z) / np.reshape(np.exp(z).sum(axis=1), (-1, 1))
    # rnk = z / z.sum(axis=1).reshape(-1, 1)
    Nk = rnk.sum(axis=0)
    # print(Nk)
    alphak = Nk / 2 + alpha0 - 1
    # betak = beta0 + (rnk * e_x_mean_lamnda_2(mk, x, sk)).sum(axis=0)
    betak = beta0 + (rnk * e_x_mean_lambda_).sum(axis=0)
    # sk = s0 + 2 * Nk * (alphak / betak)
    # mk = s0*m0 + 2 * rnk * x.reshape(-1,1) * (alpha0/beta0)
    # mk = (2 * (rnk * x.reshape(-1, 1) * (alphak / betak)).sum(axis=0) + s0 * m0) / (Nk * (alphak / betak) + s0)
    # mk = abs(m0 + Nk * 2 / s0)

    for cluster in range(k):
        for i in range(len(x)):
            if x[i] > mk[cluster]:
                test_mk_num[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * shape * abs(x[i] ** shape) / (
                    x[i])

                test_mk_den[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * abs(x[i]) ** shape * shape * (
                        shape - 1) / (2 * x[i] ** 2)
            else:
                test_mk_den[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * mk[cluster] ** (shape - 2)

                if test_mk_den[i, cluster] <= 0:
                    test_mk_den[i, cluster] = 0

                # term0 = 0
                # term1 = 0

                # if shape >=2:

                term0 = rnk[i][cluster] * e_precision_[cluster] * shape / 2 * mk[cluster] ** (
                        shape - 2) * x[i]

                term1 = rnk[i][cluster] * e_precision_[cluster] * shape / 4 * (shape - 1) * mk[
                    cluster] ** (shape - 3) * x[i] ** 2

                if term1 <= 0:
                    term1 = 0

                if term0 <= 0:
                    term0 = 0

                test_mk_num[i, cluster] = term1 + term0
                # test_mk_num[i, cluster] = rnk[i][cluster] * e_precision_[cluster] * shape / 2 * mk[cluster] ** (
                #             shape - 2) * x[i] - \
                #                           rnk[i][cluster] * e_precision_[cluster] * shape / 4 * (shape - 1) * mk[
                #                               cluster] ** (shape - 3) * x[i] ** 2

        numerator = test_mk_num.sum(axis=0)[cluster] + s0[cluster] * m0[cluster] / 2
        sk[cluster] = test_mk_den.sum(axis=0)[cluster] + s0[cluster] / 2

        mk[cluster] = numerator / sk[cluster]
    print(mk)
    gammak = gama0 + Nk
    e_ln_pi = e_ln_pi_k(gammak, Nk)
    e_ln_precision_ = digamma(alphak) - np.log(betak)
    e_precision_ = alphak / betak
    # e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, sk)

    for cluster in range(k):
        for i in range(len(x)):
            if x[i] > mk[cluster]:
                if x[i] != 0:
                    e_x_mean_lambda_[i, cluster] = x[i] ** shape - \
                                                   shape * x[i] ** shape / x[i] * mk[cluster] + \
                                                   shape / 2 * (shape - 1) * x[i] ** shape / x[i] ** 2 * (
                                                               1 / sk[cluster] +
                                                               mk[cluster] ** 2)

            else:

                t1 = -shape * x[i] * e_mean_n(sk, mk, shape - 1, k)
                e_mean2 = e_mean_n(sk, mk, shape - 2, k)

                t2 = [0, 0]
                if shape > 1:
                    t2 = shape / 2 * (shape - 1) * x[i] ** 2 * e_mean2

                e_x_mean_lambda_[i, cluster] = e_mean_n(sk, mk, shape, k)[cluster] + t1[cluster]
                e_x_mean_lambda_[i, cluster] = e_x_mean_lambda_[i, cluster] + t2[cluster]

                # e_x_mean_lambda_[i, cluster] = e_mean_n(sk, mk, shape, k)[cluster] - \
                #                                shape * x[i] * e_mean_n(sk, mk, shape - 1, k)[cluster] + \
                #                                shape / 2 * (shape - 1) * x[i] ** 2 * e_mean_n(sk, mk, shape - 2, k)[cluster]
                # if e_x_mean_lambda_[i, cluster] < 0:
                #     e_x_mean_lambda_[i, cluster] = 0

                #
                #
                # t1 = -shape * x[i] * e_mean_n(sk, mk, shape - 1, k)
                # t2 = shape / 2 * (shape - 1) * x[i] ** 2 * e_mean_n(sk, mk, shape - 2, k)
                # e_x_mean_lambda_[i, cluster] = e_mean_n(sk, mk, shape, k)[cluster]
                # if len(t1) == 2:
                #     if shape > 1:
                #         e_x_mean_lambda_[i, cluster] += t1[cluster]
                #     if shape >= 2 and len(t2) == 2:
                #         e_x_mean_lambda_[i, cluster] += t2[cluster]
                #
                # # e_x_mean_lambda_[i, cluster] = e_mean_n(sk, mk, shape, k)[cluster] - shape * x[i] * \
                # #                                e_mean_n(sk, mk, shape - 1, k)[
                # #                                    cluster] + shape / 2 * (shape - 1) * x[i] ** 2 * \
                # #                                e_mean_n(sk, mk, shape - 2, k)[cluster]
                # if e_x_mean_lambda_[i, cluster] < 0:
                #     e_x_mean_lambda_[i, cluster] = 0

            e_x_mean_lambda_[i, cluster] = abs(e_x_mean_lambda_[i, cluster])
    count += 1

    result = []
    counter = 0
    segmentedImage = np.zeros((len(x), 1), np.uint8)

    # assigning values to pixels of different segments
    rnk = list(rnk)
    rnk = [list(i) for i in rnk]
    m12 = [0, 100]
    for response in rnk:
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
