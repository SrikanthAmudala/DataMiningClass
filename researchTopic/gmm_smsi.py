import matplotlib.pyplot as plt

'''
Working GMM

'''

import random
import numpy as np
import cv2
from matplotlib import pyplot


def initialise_parameters(features, d=None, means=None, covariances=None, weights=None):
    """
    Initialises parameters: means, covariances, and mixing probabilities
    if undefined.

    Arguments:
    features -- input features data set
    """
    if not means or not covariances:
        val = 250
        n, m = features.shape
        # Shuffle features set
        indices = np.arange(n)
        np.random.shuffle(np.arange(n))
        features_shuffled = np.array([features[i] for i in indices])

        # Split into n_components subarrays
        divs = int(np.floor(n / k))
        features_split = [features_shuffled[i:i + divs] for i in range(0, n, divs)]

        # Estimate means/covariances (or both)
        if not means:
            means = []
            for i in range(k):
                rand_mean = random.randint(0, 255)
                means.append([rand_mean for j in range(d)])

        if not covariances:
            covariances = [val * np.identity(d) for i in range(k)]

    if not weights:
        weights = [float(1 / k) for i in range(k)]

    return (means, covariances, weights)


# gaussian function
def gau(mean, var, varInv, feature, d, s):
    '''

    multiply with S(xm)/Ri to the original pdf
    '''

    var_det = np.linalg.det(var)
    a = np.sqrt(2 * (np.pi ** d) * var_det)
    b = np.exp(-0.5 * np.dot((feature - mean), np.dot(varInv, (feature - mean).transpose())))
    return (s * b) / a


# calculating responsibilities
def res(likelihoods):
    tempList = []
    for comp in likelihoods:
        tempList.append(comp / sum(likelihoods))
    return tempList


# calculating likelihoods
def likeli(mean, var, varInv, weights, feature, d, s):
    temp = []
    for x in range(k):
        temp.append(weights[x] * gau(mean[x], var[x], varInv[x], feature, d, s))
    return temp


def gmm(feat, k, d, S):
    # covariances_Inv = [np.linalg.inv(covariances[0]), np.linalg.inv(covariances[1]), np.linalg.inv(covariances[2])]
    N = len(feat)
    means, covariances, weights = initialise_parameters(features=feat, d=d)
    covariances_Inv = [np.linalg.inv(covariances[i]) for i in range(k)]
    meanPrev = [np.array([0, 0, 0]) for i in range(k)]
    iteration = []
    logLikelihoods = []
    counterr = 0

    # iterating until convergence is reached
    # while sum(sum(np.absolute(np.asarray(means) - np.asarray(meanPrev)))) >= 3:

    resp = []
    likelihoods = []
    Ri = S.sum()
    for feature, s in zip(feat, S):
        classLikelihoods = likeli(means, covariances, covariances_Inv, weights, feature, d, float(s / Ri))
        rspblts = res(classLikelihoods)
        likelihoods.append(sum(classLikelihoods))
        resp.append(rspblts)

    logLikelihoods.append(sum(np.log(likelihoods)))

    nK = []
    for i in range(k):
        nK.append(sum(np.asarray(resp)[:, i:i + 1]))

    nK_weight = []
    s_resplist = []
    for i in range(k):
        s_resp = np.asarray(resp)[:, i:i + 1] * S
        # print("shape: ", s_resp)
        nK_weight.append(sum(s_resp))
        s_resplist.append(s_resp)
        # nK = [sum(np.asarray(resp)[:, 0:1]), sum(np.asarray(resp)[:, 1:2]), sum(np.asarray(resp)[:, 2:3])]
    print(s_resplist)
    print("###########\n\n")
    print(nK_weight)
    #     weights = [float(nK_weight[i] / sum(s_resplist[i])) for i in range(k)]
    #
    #     meanIterator = np.dot(np.asarray(resp).T, np.dot(feat, S.T/S.sum()))
    #     meanPrev = means
    #     # means = [meanIterator[0] / nK[0], meanIterator[1] / nK[1], meanIterator[2] / nK[2]]
    #     means = [meanIterator[i] / nK[i] for i in range(k)]
    #     counterr += 1
    #     iteration.append(counterr)
    #
    # resp = []
    # Ri = S.sum()
    # for feature, s in zip(feat, S):
    #     classLikelihoods = likeli(means, covariances, covariances_Inv, weights, feature, d, float(s/Ri))
    #     rspblts = res(classLikelihoods)
    #     resp.append(rspblts)

    return (resp, means)


def clustered_image(N, resp, means, o_shape, img):
    result = []
    counter = 0
    segmentedImage = np.zeros((N, np.shape(img)[2]), np.uint8)

    # assigning values to pixels of different segments
    for response in resp:
        maxResp = max(response)
        respmax = response.index(maxResp)
        result.append(respmax)
        segmentedImage[counter] = 255 - means[respmax]
        counter = counter + 1

    segmentedImage = segmentedImage.reshape(o_shape[0], o_shape[1], 3)
    return segmentedImage


def main(input_path, k, S):
    # reading image into matrix

    img = cv2.imread(input_path)
    pyplot.subplot(2, 1, 1)
    pyplot.imshow(img)
    o_shape = img.shape
    d = 3
    pixels = img.reshape(-1, d)

    # Total no of pixels
    N = len(pixels)
    resp, means = gmm(pixels, k, d, S)

    # segmentedImage = clustered_image(N, resp, means, o_shape, img)
    # pyplot.subplot(2, 1, 2)
    # pyplot.imshow(segmentedImage)
    # pyplot.show()


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


# input_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample.jpg"


input_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample_copy.jpg"
S = saliency(input_path)
o_S_shape = S.shape
S = S.reshape(-1)
k = 2

# main(input_path, k, S.reshape(-1))
img = cv2.imread(input_path)
pyplot.subplot(2, 1, 1)
pyplot.imshow(img)
o_shape = img.shape
d = 3
pixels = img.reshape(-1, d)
feat = pixels
# Total no of pixels
N = len(pixels)
N = len(feat)
means, covariances, weights = initialise_parameters(features=feat, d=d)
covariances_Inv = [np.linalg.inv(covariances[i]) for i in range(k)]
meanPrev = [np.array([0, 0, 0]) for i in range(k)]
iteration = []
logLikelihoods = []
counterr = 0

# iterating until convergence is reached
while sum(sum(np.absolute(np.asarray(means) - np.asarray(meanPrev)))) >= 3:
    resp = []
    likelihoods = []
    Ri = S.sum()
    saliency_list = list(S)
    for feature, s in zip(feat, saliency_list):
        classLikelihoods = likeli(means, covariances, covariances_Inv, weights, feature, d, float(s / Ri))
        rspblts = res(classLikelihoods)
        # likelihoods.append(sum(classLikelihoods))
        resp.append(rspblts)

    # logLikelihoods.append(sum(np.log(likelihoods)))

    # nK = []
    # for i in range(k):
    #     nK.append(sum(np.asarray(resp)[:, i:i + 1]))
    nK_weight = []
    # s_resplist = []

    new_mean = []
    nK= []
    den_weights = []

    for i in range(k):
        s_resp = np.asarray(resp)[:, i:i + 1]
        nK.append(s_resp)
        temp = []
        for j in range(len(resp)):
            temp.append(s_resp[j] * S[j])
        den_weights.append(temp)


    nK_np = np.asarray(nK)
    den_weights_np = np.asarray(den_weights)
    den_weights_cal_val = den_weights_np.sum(axis=1)
    weights = den_weights_cal_val/sum(den_weights_cal_val)


    resp_np = np.asarray(resp)
    mean_den = resp_np.sum(axis=0)

    mean_num = []


    for i in range(k):
        temp = []
        for s in range(len(S)):
            temp.append(resp[s][i]*(S[s]/Ri)*feat[s])
        mean_num.append(temp)

    mean_num = np.asarray(mean_num).sum(axis=1)
    mean_den = resp_np.sum(axis=0)

    for i in range(k):
        new_mean.append(mean_num/mean_den[0])
    meanPrev = means
    means = new_mean
