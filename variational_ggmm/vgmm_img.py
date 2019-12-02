from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import train_test_split
import sklearn

import pandas

import cv2

input_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/face.jpg"
img = cv2.imread(input_path)

o_shape = img.shape
k = 2
new_data = img.reshape(-1, 3)
vgmm = BayesianGaussianMixture(n_components=k)
# vgmm = GaussianMixture(n_components=k)
vgmm = vgmm.fit(new_data)
cluater = vgmm.predict(new_data)



# Reshape the input data to the orignal shape
cluater = cluater.reshape(o_shape[0], o_shape[1])
from matplotlib import pyplot
pyplot.imshow(cluater)
pyplot.show()





import matplotlib.pyplot as plt
import cv2
im = cv2.imread('/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/face.jpg')
# calculate mean value from RGB channels and flatten to 1D array
vals = im.mean(axis=2).flatten()
# plot histogram with 255 bins
b, bins, patches = plt.hist(vals, 255)
plt.xlim([0,255])
plt.show()