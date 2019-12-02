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
def gau(mean, var, varInv, feature, d):
    var_det = np.linalg.det(var)
    a = np.sqrt(2 * (np.pi ** d)*var_det)
    b = np.exp(-0.5 * np.dot((feature - mean), np.dot(varInv, (feature - mean).transpose())))
    return b / a

def covar(resp, feature, mean):
    b = np.dot(resp,( np.dot((feature - mean), (feature - mean).transpose())))


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

def gmm(feat, k, d):
    # covariances_Inv = [np.linalg.inv(covariances[0]), np.linalg.inv(covariances[1]), np.linalg.inv(covariances[2])]
    N= len(feat)
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

def main(input_path, k):
    # reading image into matrix

    img = cv2.imread(input_path)
    pyplot.subplot(2, 1, 1)
    pyplot.imshow(img)
    o_shape = img.shape
    d = 3
    pixels = img.reshape(-1, d)

    # Total no of pixels
    N = len(pixels)
    resp, means = gmm(pixels, k, d)

    segmentedImage = clustered_image(N, resp, means, o_shape, img)
    pyplot.subplot(2, 1, 2)
    pyplot.imshow(segmentedImage)
    pyplot.show()



# input_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample.jpg"
input_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample_copy.jpg"
# input_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/42049.jpg"
k = 2
main(input_path, k)

# plotting the graph of likelihood versus number of iterations

#
# pyplot.plot(iteration, logLikelihoods)
# pyplot.title('likelihood convergence')
# pyplot.ylabel('Likelihood')
# pyplot.xlabel('Iteration number')

# Show the figure.
# pyplot.show()