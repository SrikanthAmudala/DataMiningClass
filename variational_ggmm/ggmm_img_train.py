"""
Training model

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
from sklearn.model_selection import train_test_split

iris = load_iris()

X, y = iris.data, iris.target

# split data into training and test data.
train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    random_state=123)

cluster_data0 = []
cluster_data1 = []
cluster_data2 = []
for i, j in zip(train_X, train_y):
    if j == 2:
        cluster_data2.append(i)
    elif j == 1:
        cluster_data1.append(i)
    elif j == 0:
        cluster_data0.append(i)

cluster_data0 = np.asarray(cluster_data0)
cluster_data1 = np.asarray(cluster_data1)
cluster_data2 = np.asarray(cluster_data2)

# cluster_data = np.asarray([cluster_data0, cluster_data1, cluster_data2])


mean = cluster_data0.mean()
s0 = 1 / cluster_data0.var()
k = 3

temp = np.zeros((len(train_X), k))
z = []
for i, j in zip(train_y, temp):
    j[i] = 1
    z.append(j)

z = np.asarray(z)
rnk = z / z.sum(axis=0)
Nk = rnk.sum(axis=0)
shape = 1


alphak = Nk / shape + alpha0 - 1
betak = beta0 + Nk * e_x_mean_lambda_.sum(axis=0)