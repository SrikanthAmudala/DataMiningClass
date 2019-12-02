"""

SMSI GMM with gray scale Image

"""

import time

import cv2
import numpy as np
from matplotlib import pyplot
from scipy import ndimage
from sklearn.cluster import KMeans


def initialise_parameters(features, d=None, means=None, covariances=None, weights=None):
    """
    Initialises parameters: means, covariances, and mixing probabilities
    if undefined.

    Arguments:
    features -- input features data set
    """
    if not means or not covariances:
        val = 250
        n = features.shape[0]
        # Shuffle features set
        indices = np.arange(n)
        np.random.shuffle(np.arange(n))
        features_shuffled = np.array([features[i] for i in indices])

        # Split into n_components subarrays
        divs = int(np.floor(n / k))
        features_split = [features_shuffled[i:i + divs] for i in range(0, n, divs)]

        # Estimate means/covariances (or both)
        if not means:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
            means = kmeans.cluster_centers_

        if not covariances:
            covariances = [val * np.identity(d) for i in range(k)]

    if not weights:
        weights = [float(1 / k) for i in range(k)]

    return (means, covariances, weights)


# gaussian function
def gau(mean, var, varInv, feature, d):
    var_det = np.linalg.det(var)
    a = np.sqrt(2 * (np.pi ** d) * var_det)
    b = np.exp(-0.5 * np.dot((feature - mean), np.dot(varInv, (feature - mean).transpose())))
    return b / a


def covar(resp, feature, mean):
    b = np.dot(resp, (np.dot((feature - mean), (feature - mean).transpose())))


# calculating responsibilities
def res(likelihoods):
    tempList = []
    for comp in likelihoods:
        tempList.append(comp / sum(likelihoods))
    return tempList


def smsi_likeli(mean, var, varInv, s_value, feature, d):
    temp = []
    for x in range(k):
        temp.append(s_value * gau(mean[x], var[x], varInv[x], feature, d))
    return temp


def clustered_image(N, resp, means, o_shape, img):
    result = []
    counter = 0
    segmentedImage = np.zeros((N, 1), np.uint8)

    # assigning values to pixels of different segments
    for response in resp:
        maxResp = max(response)
        respmax = response.index(maxResp)
        result.append(respmax)
        segmentedImage[counter] = 255 - means[respmax]
        counter = counter + 1

    segmentedImage = segmentedImage.reshape(o_shape[0], o_shape[1])
    return segmentedImage


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
    pyplot.subplot(2, 2, 2)
    pyplot.imshow(mag)

    return mag
    # cv2.imshow('Saliency Map', mag)
    # c = cv2.waitKey(0) & 0xFF
    # if (c == 27 or c == ord('q')):
    #     cv2.destroyAllWindows()


def neighbor_prob(img, fun):
    # img = cv2.imread(img, 0)
    # fun : np.nanmean, np.nansum
    mask = np.ones((3, 3))
    # mask[1, 1] = 0
    result = ndimage.generic_filter(img, function=fun, footprint=mask, mode='constant', cval=np.NaN)

    return result


def sliding_window(im):
    rows, cols = im.shape
    final = np.zeros((rows, cols, 3, 3))
    for x in (0, 1, 2):
        for y in (0, 1, 2):
            im1 = np.vstack((im[x:], im[:x]))
            im1 = np.column_stack((im1[:, y:], im1[:, :y]))
            final[x::3, y::3] = np.swapaxes(im1.reshape(int(rows / 3), 3, int(cols / 3), -1), 1, 2)
    return final



def segmentImage(final_smsi_resp, img, N, o_shape):
    segmentedImage = np.zeros((N, np.shape(img)[2]), np.uint8)
    for i, resp in enumerate(final_smsi_resp):
        max = resp.argmax()
        segmentedImage[i] = 255 - 255 * means[max]
    segmentedImage = segmentedImage.reshape(o_shape[0], o_shape[1], 3)
    pyplot.subplot(1,1,1)
    pyplot.imshow(segmentedImage)
    pyplot.show()

# input_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample.jpg"
input_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample_copy.jpg"

# img = io.imread("https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/color/135069.jpg", True)*255
# print(img*255)

img = cv2.imread(input_path)
pyplot.subplot(2, 2, 1)
pyplot.imshow(img)

k = 2
o_shape = img.shape
d = 3
pixels = img.reshape(-1, d)

# Total no of pixels
N = len(pixels)
# resp, means = gmm(pixels, k, d)
feat = pixels

# covariances_Inv = [np.linalg.inv(covariances[0]), np.linalg.inv(covariances[1]), np.linalg.inv(covariances[2])]

# s = saliency(input_path)
s = saliency(input_path)

N = len(feat)

means, covariances, weights = initialise_parameters(features=feat, d=d)

# print("Init: \n"
#       "Means: ", means, "\ncovar: ", covariances, "\n weights: ", weights)

covariances_Inv = [np.linalg.inv(covariances[i]) for i in range(k)]

# means = np.asarray([[269.41929646,329.31834008,179.93338807],
#  [284.57160791 ,418.44757094 ,279.60823943]])


# meanPrev = [np.array([0, 0, 0]) for i in range(k)]
meanPrev = np.zeros((k, d))

# meanPrev = np.asarray([[269.41929646, 329.31834008 ,179.93338807],
#  [284.72743147 ,418.65345343 ,279.72591555]])

counterr = 0

smsi_init_weights = np.ones((o_shape[0] * o_shape[1], k))
R_value = neighbor_prob(s, np.nansum).reshape(-1)
s_value = s.reshape(-1)

while abs((np.asarray(meanPrev) - np.asarray(means)).sum()) > 0.1:
    start = time.time()
    """
    Responsibilities 
    """
    smsi_resp = []
    for i, feature in enumerate(feat):
        # s * p(x|0)
        smsi_classLikelihoods = smsi_likeli(mean=means, var=covariances, varInv=covariances_Inv, s_value=s_value[i],
                                            feature=feature, d=d)
        smsi_resp.append(smsi_classLikelihoods)

    smsi_resp = np.asarray(smsi_resp)

    final_resp = []
    for cluster in range(k):
        # Sum of neighbour pixels
        result = ndimage.generic_filter(smsi_resp[:, cluster].reshape(o_shape[0], o_shape[1]), np.nansum,
                                        footprint=np.ones((3, 3)),
                                        mode='constant', cval=np.NaN).reshape(-1)
        final_resp.append(result * (smsi_init_weights[:, cluster] / R_value))

    # numerator of gama
    smsi_resp_num = np.asarray(final_resp).T

    # denominator of gama
    final_resp_den = smsi_resp_num.sum(axis=1)

    # gama value of the smsi

    final_smsi_resp = np.asarray([smsi_resp_num[:, i] / final_resp_den for i in range(k)]).T



    # plot

    segmentImage(final_smsi_resp, img, N, o_shape)



    """
    SMSI MEAN
    """

    smsi_mean_den = final_smsi_resp.sum(axis=0)

    #
    smsi_mean_num_s_x = np.asarray([s_value * feat[:, i] for i in range(d)]).T
    smsi_mean_num = []
    smsi_mean_num_s_x = ndimage.generic_filter(smsi_mean_num_s_x.reshape(o_shape[0], o_shape[1], d), np.nansum,
                                               footprint=np.ones((3, 3, 3)), mode='constant', cval=np.NaN).reshape(-1, d)


    for j in range(k):
        smsi_mean_num.append([(smsi_mean_num_s_x[:, i] / R_value) * final_smsi_resp[:, j] for i in range(d)])

    smsi_mean_num = np.asarray(smsi_mean_num).T
    smsi_mean = (smsi_mean_num.sum(axis=0) / smsi_mean_den).T
    meanPrev = means
    means = smsi_mean

    """
    CO VAR
    """

    """
    # f_u = np.asarray([feat - means[i] for i in range(k)]).T

    f_u = []
    for j in range(k):
        f_u.append(feat - smsi_mean[j])
    f_u = np.asarray(f_u)

    f_u_1 = [f_u[i] - smsi_mean[i] for i in range(k)]
    f_u_1 = np.asarray(f_u_1)

    f_u = []

    # (x - u) * (x - u)' # for each pixel the co var becomes 3x3
    #  s * (x - u) * (x - u)' = N*k*d*d
    for i in range(k):
        f_u.append([f_u_1[i][j].reshape(3, 1) * f_u_1[i][j].reshape(1, 3) * s_value[j] for j in range(len(feat))])

    f_u = np.asarray(f_u)
    s_f_u = f_u.reshape(len(feat), k, d, d)

    # Sum of Neighbors of all the Co Var Matrix for both the clusters

    covar_num_s_x_u = np.asarray(
        [ndimage.generic_filter(s_f_u[:, i].reshape(o_shape[0], o_shape[1], 3, 3), np.nansum,
                                footprint=np.ones((3, 3, 3, 3)), mode='constant', cval=np.NaN).reshape(-1) for i in
         range(k)]).T

    covar_num_s_x_u = covar_num_s_x_u.reshape(s_f_u.shape)

    test1 = []
    for j in range(k):
        test1.append([smsi_resp[i][j] * covar_num_s_x_u[i][j] / R_value[i] for i in range(len(feat))])

    test1 = np.asarray(test1)
    smsi_covar_num = test1.sum(axis=1)
    smsi_covar = [smsi_covar_num[i] / smsi_mean_den[i] for i in range(k)]
    covariances = smsi_covar
    covariances_Inv = [np.linalg.inv(covariances[i]) for i in range(k)]

    # covariances = [covar[i] * np.identity(d) for i in range(k)]
    # covariances_Inv = [np.linalg.inv(covariances[i]) for i in range(k)]
    """

    """
    WEIGHTS
    """

    # s*gama
    """
    resp_s = np.asarray([final_smsi_resp[:, i] * s_value for i in range(k)]).T
    # temp = []
    # for j in range(k):
    #     temp.append([resp_s[i][j] * s_value[i] for i in range(len(feat))])
    # temp = np.asarray(temp).T

    w_numerator = np.asarray([ndimage.generic_filter(resp_s[:, i].reshape(o_shape[0], o_shape[1]), np.nansum,
                                                     footprint=np.ones((3, 3)), mode='constant', cval=np.NaN).reshape(
        -1)
        for i in range(k)]).T

    w_den = w_numerator.sum(axis=1)

    smsi_weights = np.asarray([w_numerator[:, i] / w_den for i in range(k)]).T

    smsi_init_weights = smsi_weights
    condition = abs((np.asarray(meanPrev) - np.asarray(means)).sum())
    # print("Cov: \n", covariances)
    # print(counterr, condition, " Means: ", smsi_mean, "Time: ", time.time() - start)
    print(counterr, " Means: ", smsi_mean)
    counterr += 1
    """







# segmentedImage = np.zeros((N, np.shape(img)[2]), np.uint8)
# for i, resp in enumerate(final_smsi_resp):
#     max = resp.argmax()
#     segmentedImage[i] = 255 - 255 * means[max]
# segmentedImage = segmentedImage.reshape(o_shape[0], o_shape[1], 3)
# pyplot.subplot(2, 2, 3)
# pyplot.imshow(segmentedImage)
# pyplot.show()