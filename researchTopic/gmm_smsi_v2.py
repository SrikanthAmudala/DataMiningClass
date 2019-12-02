'''
Working GMM

'''
import random
import glob, os
from PIL import Image
import numpy as np
import cv2
import PIL
from matplotlib import pyplot
from scipy import ndimage


def initialise_parameters(features, d=None, means=None, covariances=None, weights=None):
    """
    Initialises parameters: means, covariances, and mixing probabilities
    if undefined.

    Arguments:
    features -- input features data set
    """
    n, m = features.shape
    if not means or not covariances:
        val = 250

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

        weights = np.asarray([float(1 / k) for i in range(k)])

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


# calculating likelihoods
def likeli(mean, var, varInv, weights, feature, d):
    temp = []
    for x in range(k):
        temp.append(weights[x] * gau(mean[x], var[x], varInv[x], feature, d))
    return temp


def smsi_likeli(mean, var, varInv, S_value, feature, d):
    temp = []
    for x in range(k):
        temp.append(S_value * gau(mean[x], var[x], varInv[x], feature, d))
    return temp

def gmm(feat, k, d):
    # covariances_Inv = [np.linalg.inv(covariances[0]), np.linalg.inv(covariances[1]), np.linalg.inv(covariances[2])]
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
        for feature in feat:
            classLikelihoods = likeli(means, covariances, covariances_Inv, weights, feature, d)
            rspblts = res(classLikelihoods)
            likelihoods.append(sum(classLikelihoods))
            resp.append(rspblts)

        logLikelihoods.append(sum(np.log(likelihoods)))

        nK = []
        for i in range(k):
            nK.append(sum(np.asarray(resp)[:, i:i + 1]))

        # nK = [sum(np.asarray(resp)[:, 0:1]), sum(np.asarray(resp)[:, 1:2]), sum(np.asarray(resp)[:, 2:3])]

        weights = [float(nK[i] / N) for i in range(k)]
        meanIterator = np.dot(np.asarray(resp).T, feat)

        # covarIterator = np.dot(np.asarray(resp).T, np.dot( feat-means, (feat - means).transpose()))

        # np.dot((feature - mean), np.dot(varInv, (feature - mean).transpose()))

        meanPrev = means
        # means = [meanIterator[0] / nK[0], meanIterator[1] / nK[1], meanIterator[2] / nK[2]]
        means = [meanIterator[i] / nK[i] for i in range(k)]
        counterr += 1
        iteration.append(counterr)

    resp = []

    for feature in feat:
        classLikelihoods = likeli(means, covariances, covariances_Inv, weights, feature, d)
        rspblts = res(classLikelihoods)
        resp.append(rspblts)

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
    pyplot.subplot(2, 2, 1)
    pyplot.imshow(mag)

    return mag
    # cv2.imshow('Saliency Map', mag)
    # c = cv2.waitKey(0) & 0xFF
    # if (c == 27 or c == ord('q')):
    #     cv2.destroyAllWindows()


def gmm_test(feat, d):
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
        for feature in feat:
            classLikelihoods = likeli(means, covariances, covariances_Inv, weights, feature, d)
            rspblts = res(classLikelihoods)
            likelihoods.append(sum(classLikelihoods))
            resp.append(rspblts)

        logLikelihoods.append(sum(np.log(likelihoods)))

        nK = []
        for i in range(k):
            nK.append(sum(np.asarray(resp)[:, i:i + 1]))

        # nK = [sum(np.asarray(resp)[:, 0:1]), sum(np.asarray(resp)[:, 1:2]), sum(np.asarray(resp)[:, 2:3])]

        weights = [float(nK[i] / N) for i in range(k)]
        meanIterator = np.dot(np.asarray(resp).T, feat)

        # covarIterator = np.dot(np.asarray(resp).T, np.dot( feat-means, (feat - means).transpose()))

        # np.dot((feature - mean), np.dot(varInv, (feature - mean).transpose()))
        meanPrev = means
        # means = [meanIterator[0] / nK[0], meanIterator[1] / nK[1], meanIterator[2] / nK[2]]
        means = [meanIterator[i] / nK[i] for i in range(k)]
        counterr += 1
        iteration.append(counterr)

    resp = []

    for feature in feat:
        classLikelihoods = likeli(means, covariances, covariances_Inv, weights, feature, d)
        rspblts = res(classLikelihoods)
        resp.append(rspblts)

    # return (resp, means)

    segmentedImage = clustered_image(N, resp, means, o_shape, img)
    pyplot.subplot(2, 1, 2)
    pyplot.imshow(segmentedImage)
    pyplot.show()

def neighbor_prob(img, fun):

    # img = cv2.imread(img, 0)
    # fun : np.nanmean, np.nansum
    mask = np.ones((3, 3))
    # mask[1, 1] = 0
    result = ndimage.generic_filter(img, function=fun, footprint=mask, mode='constant', cval=np.NaN)

    return result

def sliding_window(rows, cols, im):
    final = np.zeros((rows, cols, 3, 3))
    for x in (0,1,2):
        for y in (0,1,2):
            im1 = np.vstack((im[x:],im[:x]))
            im1 = np.column_stack((im1[:,y:],im1[:,:y]))
            final[x::3,y::3] = np.swapaxes(im1.reshape((rows/3),3,int(cols/3),-1),1,2)
    return final


# input_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample.jpg"
input_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample_copy.jpg"
# input_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/42049.jpg"
k = 2

img = cv2.imread(input_path)
pyplot.subplot(2, 1, 1)
pyplot.imshow(img)
o_shape = img.shape
d = 3
pixels = img.reshape(-1, d)

# Total no of pixels
N = len(pixels)
# resp, means = gmm(pixels, k, d)
feat = pixels

# covariances_Inv = [np.linalg.inv(covariances[0]), np.linalg.inv(covariances[1]), np.linalg.inv(covariances[2])]

s = saliency(input_path)
# s = s.reshape(-1)
N = len(feat)
means, covariances, weights = initialise_parameters(features=feat, d=d)
covariances_Inv = [np.linalg.inv(covariances[i]) for i in range(k)]

meanPrev = [np.array([0, 0, 0]) for i in range(k)]
iteration = []
logLikelihoods = []
counterr = 0
# R_value = []
R_value = neighbor_prob(s, np.nansum)
R_value = R_value.reshape(-1)
s_value = s.reshape(-1)
smsi_init_weights = np.ones((o_shape[0]*o_shape[1], k))
# iterating until convergence is reached
while sum(sum(np.absolute(np.asarray(means) - np.asarray(meanPrev)))) >= 3:
    smsi_resp = []

    for i, feature in enumerate(feat):
        smsi_classLikelihoods = smsi_likeli(means, covariances, covariances_Inv, s_value[i], feature, d)
        smsi_resp.append(smsi_classLikelihoods)

    smsi_resp = np.asarray(smsi_resp)

    final_resp = []
    for cluster in range(k):
        test = smsi_resp[:, cluster]
        result = ndimage.generic_filter(test.reshape(o_shape[0], o_shape[1]), np.nansum, footprint=np.ones((3, 3)), mode='constant', cval=np.NaN).reshape(-1)
        final_resp.append(result*smsi_init_weights[:,cluster]/R_value)

    # numerator of gama
    smsi_resp_num = np.asarray(final_resp).T

    # denominator of gama
    final_resp_den = smsi_resp_num.sum(axis=1)

    # gama value of the smsi
    final_smsi_resp = []
    for cluster in range(k):
        final_smsi_resp.append(smsi_resp_num[:,cluster]/final_resp_den)

    final_smsi_resp = np.asarray(final_smsi_resp).T


    # # weights for smsi
    # smsi_weigths = []
    #
    # # smsi_weights_num = [final_smsi_resp[:, i] * s for i in range(k)]
    # smsi_weights_num = []
    # for i in range(k):
    #     smsi_weights_num_temp = final_smsi_resp[:, i] * s.reshape(-1)
    #     temp = ndimage.generic_filter(smsi_weights_num_temp, np.nansum, 3, mode='constant', cval=np.NaN).reshape(-1)
    #     smsi_weights_num.append(temp)
    #
    # smsi_weights_num = np.asarray(smsi_weights_num).T
    # # smsi_weights = smsi_weights_num/smsi_weights_num.sum(axis=1)
    # smsi_weights = [smsi_weights_num[:, i] / smsi_weights_num.sum(axis=0).sum() for i in range(k)]
    # smsi_weights = np.asarray(smsi_weights).T
    # smsi_init_weights = smsi_weights


    smsi_mean_den = final_smsi_resp.sum(axis=0)

    # s_mean = s.reshape(-1)
    meanPrev = means
    smsi_mean_num_s_x = np.asarray([feat[:, i] * s_value for i in range(d)]).T
    smsi_mean_num = []

    for i in range(k):
        for j in range(d):
            smsi_mean_num_s_x[:,j] = ndimage.generic_filter(smsi_mean_num_s_x[:,j].reshape(o_shape[0], o_shape[1]), np.nansum, footprint=np.ones((3,3)), mode='constant', cval=np.NaN).reshape(-1)
            smsi_mean_num_s_x[:, j] = smsi_mean_num_s_x[:,j]/R_value
        smsi_mean_num.append(smsi_mean_num_s_x)

    smsi_mean_num = np.asarray(smsi_mean_num).reshape(o_shape[0]*o_shape[1], o_shape[-1], k)
    smsi_mean = (smsi_mean_num.sum(axis=0)/smsi_mean_den).T

    print("smsi_mean den: ",smsi_mean_den)
    means = smsi_mean
    print("Means: ", means)
    # print("Weights: ", smsi_weights)
    # logLikelihoods.append(sum(np.log(likelihoods)))

    # sum of all the resp of each cluster
    nK = []

    # for i in range(k):
    #     nK.append(sum(np.asarray(resp)[:, i:i + 1]))

    # nK = [sum(np.asarray(resp)[:, 0:1]), sum(np.asarray(resp)[:, 1:2]), sum(np.asarray(resp)[:, 2:3])]

    # weights = [float(nK[i] / N) for i in range(k)]
    '''
    meanIterator = np.dot(np.asarray(resp).T, feat)

    # covarIterator = np.dot(np.asarray(resp).T, np.dot( feat-means, (feat - means).transpose()))
    
    # np.dot((feature - mean), np.dot(varInv, (feature - mean).transpose()))
    meanPrev = means
    # means = [meanIterator[0] / nK[0], meanIterator[1] / nK[1], meanIterator[2] / nK[2]]
    means = [meanIterator[i] / nK[i] for i in range(k)]
    '''
    counterr += 1
    iteration.append(counterr)
'''
resp = []

for feature in feat:
    classLikelihoods = likeli(means, covariances, covariances_Inv, weights, feature, d)
    rspblts = res(classLikelihoods)
    resp.append(rspblts)

# return (resp, means)

segmentedImage = clustered_image(N, resp, means, o_shape, img)
pyplot.subplot(2, 1, 2)
pyplot.imshow(segmentedImage)
pyplot.show()
'''