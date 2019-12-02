"""
brest cancer with k fold: accuracy: 0.73
"""

from sklearn.datasets import load_iris
import sklearn
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict
from scipy.special import digamma
from scipy.special import gamma
import math
import mpmath
import sys

from sklearn.model_selection import StratifiedKFold

from scipy import special


# E[ln pik]
def e_ln_pi_k(gama0, Nk):
    gammak = gama0 + Nk
    return digamma(gammak) - digamma(gammak.sum())


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


# iris = load_iris()
iris = sklearn.datasets.load_breast_cancer()
k = 2

# X = iris.data[:, :4]  # we only take the first two features.

X = iris.data[:]
y = iris.target

# import pandas
#
# data = pandas.read_csv("/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/classification/heart.csv", header=None,
#                        skiprows=1)
#
# X = data.iloc[:, 0:13]  # slicing: all rows and 1 to 4 cols
# # store response vector in "y"
# y = data.iloc[:, 13]
# k = 2


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

skf = StratifiedKFold(n_splits=4)
skf.get_n_splits(X, y)
accuracy_list = []
for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    print("####")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    d = X_train.shape[-1]

    z_list = []
    alpthak_list = []
    betak_list = []
    mk_list = []
    gammak_list = []
    sk_list = []

    for dim in range(d):
        x = X_train[:, dim]
        o_shape = x.shape
        x = x.reshape(-1)
        # z = y_train
        temp = np.zeros((len(y_train), k))
        z = []
        for i, j in zip(y_train, temp):
            j[i] = 1
            z.append(j)
        z = np.asarray(z)
        rnk = np.exp(z) / np.reshape(np.exp(z).sum(axis=1), (-1, 1))
        Nk = rnk.sum(axis=0)
        alpha0 = np.asarray([x.mean() ** 2 / x.var() for _ in range(k)])
        beta0 = np.asarray([x.mean() / i for i in alpha0])

        clustered_img = defaultdict(list)
        for v, v1 in zip(x, y_train): clustered_img[v1].append(v)

        gamma0 = []

        for i in clustered_img.keys():
            gamma0.append(len(clustered_img.get(i)) / len(x))
        gamma0 = np.asarray(gamma0)

        variance = []
        for i in clustered_img.keys():
            variance.append(np.var(np.asarray(clustered_img.get(i))))

        sk = np.asarray(variance)
        sk_list.append(sk)
        m0 = []
        for i in clustered_img.keys():
            m0.append(np.asarray(clustered_img.get(i)).mean())
        mk = np.asarray(m0)
        mk_list.append(mk)

        test_mk_num = z.copy()
        test_mk_den = z.copy()

        e_x_mean_lambda_ = z.copy()
        shape = np.asarray([2 for _ in range(k)])

        for cluster in range(k):
            for i in range(len(x)):
                if x[i] > mk[cluster]:
                    if x[i] != 0:
                        e_x_mean_lambda_[i, cluster] = x[i] ** shape[cluster] - shape[cluster] * x[i] ** shape[
                            cluster] / x[
                                                           i] * \
                                                       mk[
                                                           cluster] + shape[cluster] / 2 * (
                                                               shape[cluster] - 1) * x[i] ** shape[cluster] / x[
                                                           i] ** 2 * (
                                                               1 / sk[cluster] + mk[cluster] ** 2)

                else:

                    t1 = -shape[cluster] * x[i] * e_mean_n(sk, mk, shape[cluster] - 1, k)
                    e_mean2 = e_mean_n(sk, mk, shape[cluster] - 2, k)
                    t2 = [0, 0]

                    if shape[cluster] > 1:
                        t2 = shape[cluster] / 2 * (shape[cluster] - 1) * x[i] ** 2 * e_mean2

                    e_x_mean_lambda_[i, cluster] = abs(
                        e_mean_n(sk, mk, shape[cluster], k)[cluster] + t1[cluster] + t2[cluster])

                    # e_x_mean_lambda_[i, cluster] = abs(e_x_mean_lambda_[i, cluster])

        alphak = Nk / 2 + alpha0 - 1
        alpthak_list.append(alphak)

        betak = beta0 + (rnk * e_x_mean_lambda_).sum(axis=0)
        betak_list.append(betak)

        gammak = gamma0 + Nk
        gammak_list.append(gammak)

        # e_ln_pi = e_ln_pi_k(gammak, Nk)

        # for i in range(k):
        #     p1 = e_ln_pi[i] + (1 / shape[i]) * e_ln_precision_[i] + np.log(shape[i]) - np.log(2 * gamma(1 / shape[i]))
        #     z[:, i] = (p1 - (e_precision_[i] * e_x_mean_lambda_[:, i]).reshape(-1, 1)).reshape(-1)
        # z_list.append(z)

        #
        # z_list = np.asarray(z_list)
        # z_final = z_list.sum(axis = 0)
        #
        # alpthak_list = np.asarray(alpthak_list)
        # betak_list = np.asarray(betak_list)
        # gammak_list = np.asarray(gammak_list)
        # mk_list = np.asarray(mk_list)
        # sk_list = np.asarray(sk_list)

        e_ln_pi = e_ln_pi_k(gammak, Nk)
        e_ln_precision_ = digamma(alphak) - np.log(betak)
        e_precision_ = alphak / betak

        L1 = lowerbound_first_dir(rnk, x, shape, mk, sk, e_precision_).sum(axis=0)
        L2 = lowerbound_second_dir(rnk, x, shape, mk, sk, e_precision_).sum(axis=0)
        L1 += sys.float_info.epsilon
        L2 += sys.float_info.epsilon
        delta_lowerbound = L1 / L2
        delta_lowerbound += sys.float_info.epsilon
        shape = shape - 0.01 * (delta_lowerbound)
        x_test = X_test[:, dim]

        temp1 = np.zeros((len(x_test), k))
        z1 = []

        for i, j in zip(y_test, temp1):
            j[i] = 1
            z1.append(j)

        z1 = np.asarray(z1)

        e_x_mean_lambda_ = z1.copy()

        for cluster in range(k):
            for i in range(len(x_test)):
                if x_test[i] > mk[cluster]:
                    if x_test[i] != 0:
                        e_x_mean_lambda_[i, cluster] = x_test[i] ** shape[cluster] - \
                                                       shape[cluster] * x_test[i] ** shape[cluster] / x_test[i] * mk[
                                                           cluster] + \
                                                       shape[cluster] / 2 * (shape[cluster] - 1) * x_test[i] ** shape[
                                                           cluster] / \
                                                       x_test[i] ** 2 * (
                                                               1 / sk[cluster] +
                                                               mk[cluster] ** 2)

                else:

                    t1 = -shape[cluster] * x_test[i] * e_mean_n(sk, mk, shape[cluster] - 1, k)
                    e_mean2 = e_mean_n(sk, mk, shape[cluster] - 2, k)

                    t2 = [0, 0]
                    if shape[cluster] > 1:
                        t2 = shape[cluster] / 2 * (shape[cluster] - 1) * x_test[i] ** 2 * e_mean2

                    e_x_mean_lambda_[i, cluster] = e_mean_n(sk, mk, shape[cluster], k)[cluster] + t1[cluster] + t2[
                        cluster]

                e_x_mean_lambda_[i, cluster] = abs(e_x_mean_lambda_[i, cluster])

        for i in range(k):
            p1 = e_ln_pi[i] + (1 / shape[i]) * e_ln_precision_[i] + np.log(shape[i]) - np.log(2 * gamma(1 / shape[i]))
            z1[:, i] = (p1 - (e_precision_[i] * e_x_mean_lambda_[:, i]).reshape(-1, 1)).reshape(-1)

        rnk = np.exp(z1) / np.reshape(np.exp(z1).sum(axis=1), (-1, 1))
        z_list.append(rnk)

    z_list = np.asarray(z_list).sum(axis=0)

    rnk = list(z_list)
    rnk = [list(i) for i in rnk]
    result = []

    for response in rnk:
        maxResp = max(response)
        respmax = response.index(maxResp)
        result.append(respmax)

    result = np.asarray(result)
    accuracy = metrics.accuracy_score(y_test, result)

    print("Accuracy: ", accuracy)
    accuracy_list.append(accuracy)

accuracy_list = np.asarray(accuracy_list)
print("Final Accuracy: ", accuracy_list.mean())
