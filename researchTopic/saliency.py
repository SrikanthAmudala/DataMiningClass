import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn import mixture
import pandas

WIDTH = 128  # has a great influence on the result


def saliency(input_path):
    img = cv2.imread(input_path, 0)
    # img = cv2.resize(img, (WIDTH, int(WIDTH * img.shape[0] / img.shape[1])))

    c = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = np.sqrt(c[:, :, 0] ** 2 + c[:, :, 1] ** 2)
    spectralResidual = np.exp(np.log(mag) - cv2.boxFilter(np.log(mag), -1, (3, 3)))

    c[:, :, 0] = c[:, :, 0] * spectralResidual / mag
    c[:, :, 1] = c[:, :, 1] * spectralResidual / mag
    c = cv2.dft(c, flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE))
    mag = c[:, :, 0] ** 2 + c[:, :, 1] ** 2
    cv2.normalize(cv2.GaussianBlur(mag, (9, 9), 3, 3), mag, 0., 1., cv2.NORM_MINMAX)
    plt.subplot(2, 2, 1)
    plt.imshow(mag)

    return mag
    # cv2.imshow('Saliency Map', mag)
    # c = cv2.waitKey(0) & 0xFF
    # if (c == 27 or c == ord('q')):
    #     cv2.destroyAllWindows()


input_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample.jpg"

X = saliency(input_path)
df = pandas.DataFrame(X)
print(df[0].max())

print(X.shape)

gmm = mixture.GaussianMixture(covariance_type='full', n_components=2)

img = cv2.imread(input_path, 0)
plt.subplot(2,2,3)
plt.imshow(img)
o_shape = img.shape
img = img.reshape(-1, 1)
gmm.fit(img)
clusters = gmm.predict(img)
print(clusters.shape)

img = clusters.reshape(o_shape[0], o_shape[-1])
plt.subplot(2, 2, 2)
plt.imshow(img)
plt.show()

# print(X)
#
# gmm = mixture.GMM(covariance_type='full', n_components=2,weights_="")
# gmm.fit(X)


# X = X.reshape(-1, 3)
#
# print("After Reshape",X.shape)
#
# clusters = gmm.predict(X)
# print("cluster shape: ",clusters.shape)
