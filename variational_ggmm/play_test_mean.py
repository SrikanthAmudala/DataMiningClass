"""
shape = 2
"""

import math
from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot
from scipy.special import digamma
from scipy.special import gamma
from sklearn.cluster import KMeans


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
k = 2
alpha0 = np.asarray([x.mean() ** 2 / x.var() for _ in range(k)])
beta0 = np.asarray([x.mean() / i for i in alpha0])
# beta0 = np.asarray([(1e-20) * 1 / i for i in alpha0])
shape = 2

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

# sk = s0
mk = (2 * (rnk * x.reshape(-1, 1) * (alpha0 / beta0)).sum(axis=0) + s0 * m0) / (Nk * (alpha0 / beta0) + s0)



gammak = gama0 + Nk
alphak = Nk / 2 + alpha0 - 1
betak = beta0 + (rnk * e_x_mean_lamnda_2(mk, x, sk)).sum(axis=0)
e_ln_pi = e_ln_pi_k(gammak, Nk)
e_ln_precision_ = digamma(alphak) - np.log(betak)
e_precision_ = alphak / betak
e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, sk)

count = 0

while (count < 100):
    for i in range(k):
        p1 = e_ln_pi[i] + (1 / shape) * e_ln_precision_[i] + np.log(shape) - np.log(2 * gamma(1 / shape))
        z[:, i] = (p1 - (e_precision_[i] * e_x_mean_lambda_[:, i]).reshape(-1, 1)).reshape(-1)
    rnk = np.exp(z) / np.reshape(np.exp(z).sum(axis=1), (-1, 1))
    # rnk = z / z.sum(axis=1).reshape(-1, 1)
    Nk = rnk.sum(axis=0)
    # print(Nk)
    alphak = Nk / 2 + alpha0 - 1
    betak = beta0 + (rnk * e_x_mean_lamnda_2(mk, x, sk)).sum(axis=0)
    sk = s0 + 2 * Nk * (alphak / betak)
    # mk = s0*m0 + 2 * rnk * x.reshape(-1,1) * (alpha0/beta0)
    mk = (2 * (rnk * x.reshape(-1, 1) * (alphak / betak)).sum(axis=0) + s0 * m0) / (Nk * (alphak / betak) + s0)
    # mk = abs(m0 + Nk * 2 / s0)

    print(mk)
    gammak = gama0 + Nk
    e_ln_pi = e_ln_pi_k(gammak, Nk)
    e_ln_precision_ = digamma(alphak) - np.log(betak)
    e_precision_ = alphak / betak
    e_x_mean_lambda_ = e_x_mean_lamnda_2(mk, x, sk)
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
# pyplot.savefig("/Users/Srikanth/PycharmProjects/DataMiningClass/outputs/eagle/" + str(count) + ".png")
pyplot.show()
