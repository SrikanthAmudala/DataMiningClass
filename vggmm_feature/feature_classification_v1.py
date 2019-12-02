"""

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


import pandas

data = pandas.read_csv("/home/k_mathin/PycharmProjects/DataMiningClass/datasets/classification/pulsar_stars.csv",
                       header=None,
                       skiprows=1)

X = data.iloc[:, 0:7]  # slicing: all rows and 1 to 4 cols
# store response vector in "y"
y = data.iloc[:, 8]
k = 2
X = np.asarray(X)
y = np.asarray(y)

skf = StratifiedKFold(n_splits=4)
skf.get_n_splits(X, y)
accuracy_list = []

for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    print("####")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

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
        gammak = gamma0 + Nk
        alphak = Nk / 2 + alpha0 - 1
        betak = beta0 + (rnk * e_x_mean_lambda_).sum(axis=0)

        e_ln_pi = e_ln_pi_k(gammak, Nk)
        e_ln_precision_ = digamma(alphak) - np.log(betak)
        e_precision_ = alphak / betak

        # Feature
        term1 = (rnk * (digamma(alphak) - np.log(betak))).sum(axis=1) / 2
        term2 = 1 / 2 * (rnk * (alphak / betak) * ((x.reshape(-1, 1) - mk.reshape(-1, 1).T) ** 2 + 1 / sk)).sum(axis=1)

        row_in_e = np.exp(term1 - term2)
        w = np.asarray([1 for _ in range(k)])

        epsolon = mk
        var_test = sk
        epsolon_in = np.exp(
            -1 / 2 * 1 / var_test * ((x.reshape(-1, 1) - epsolon.reshape(-1, 1).T) ** 2) + 1 / 2 * np.log(1 / var_test))

        num1 = w * row_in_e.reshape(-1, 1)
        den1 = w * row_in_e.reshape(-1, 1) + (1 - w) * epsolon_in
        fik = np.divide(num1, den1, out=np.zeros_like(num1), where=den1 != 0)
        # fik = (w * row_in_e.reshape(-1, 1)) / (w * row_in_e.reshape(-1, 1) + (1 - w) * epsolon_in)

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
                    test_mk_den[i, cluster] = fik[i][cluster] * rnk[i][cluster] * e_precision_[cluster] * mk[
                        cluster] ** (
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

            numerator = test_mk_num.sum(axis=0)[cluster] + sk[cluster] * m0[cluster] / 2
            sk[cluster] = test_mk_den.sum(axis=0)[cluster] + sk[cluster] / 2

            mk[cluster] = numerator / sk[cluster]

        # alphak = Nk / 2 + alpha0 - 1

        # betak = beta0 + (rnk * e_x_mean_lambda_).sum(axis=0)

        alphak = (rnk * fik).sum(axis=0) / 2 + alpha0 - 1
        betak = beta0 + (rnk * fik * e_x_mean_lambda_).sum(axis=0)

        alpthak_list.append(alphak)
        betak_list.append(betak)

        gammak = gamma0 + Nk
        gammak_list.append(gammak)

        # L1 = lowerbound_first_dir(rnk, x, shape, mk, sk, e_precision_).sum(axis=0)
        # L2 = lowerbound_second_dir(rnk, x, shape, mk, sk, e_precision_).sum(axis=0)
        # L1 += sys.float_info.epsilon
        # L2 += sys.float_info.epsilon
        # delta_lowerbound = L1 / L2
        # delta_lowerbound += sys.float_info.epsilon
        # shape = shape - 0.01 * (delta_lowerbound)
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

        w = fik.sum(axis=0) / len(fik)

        for i in range(k):
            p1 = e_ln_pi[i] + (1 / shape[i]) * e_ln_precision_[i] + np.log(shape[i]) - np.log(2 * gamma(1 / shape[i]))
            z1[:, i] = (p1 - (e_precision_[i] * e_x_mean_lambda_[:, i]).reshape(-1, 1)).reshape(-1)



        rnk = np.exp(z1) / np.reshape(np.exp(z1).sum(axis=0), (-1, 1)).T



        term1 = (rnk * (digamma(alphak) - np.log(betak))).sum(axis=1) / 2

        term2 = 1 / 2 * (rnk * (alphak / betak) * ((x_test.reshape(-1, 1) - mk.reshape(-1, 1).T) ** 2 + 1 / sk)).sum(
            axis=1)
        row_in_e = np.exp(term1 - term2)

        epsolon_in = np.exp(
            -1 / 2 * 1 / var_test * ((x_test.reshape(-1, 1) - epsolon.reshape(-1, 1).T) ** 2) + 1 / 2 * np.log(
                1 / var_test))
        num1 = w * row_in_e.reshape(-1, 1)
        den1 = w * row_in_e.reshape(-1, 1) + (1 - w) * epsolon_in
        fik = np.divide(num1, den1, out=np.zeros_like(num1), where=den1 != 0)
        # fik = (w * row_in_e.reshape(-1, 1)) / (w * row_in_e.reshape(-1, 1) + (1 - w) * epsolon_in)

        epsolon = (fik * x_test.reshape(-1, 1)).sum(axis=0) / fik.sum(axis=0)
        var_test = (fik * ((x_test - epsolon.reshape(-1, 1)) ** 2).T).sum(axis=0) / fik.sum(axis=0)

        for i in range(k):
            p1 = (1 / shape[i]) * e_ln_precision_[i] + np.log(shape[i]) - np.log(2 * gamma(1 / shape[i]))
            p2 = fik[:, i] * (p1 - (e_precision_[i] * e_x_mean_lambda_[:, i]).reshape(-1, 1)).reshape(-1)
            p3 = (1 / 2) * np.log(1 / var_test[i]) + np.log(2) - np.log(2 * gamma(1 / 2)) - 1 / var_test[i] * (
                    x_test - epsolon[i]) ** 2
            p4 = e_ln_pi[i] + p2 + (1 - fik[:, i]) * p3
            z1[:, i] = p4

        # np.seterr(divide='ignore', invalid='ignore')
        rnk = np.exp(z1) / np.reshape(np.exp(z1).sum(axis=0), (-1, 1)).T
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
