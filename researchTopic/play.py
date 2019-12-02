from sklearn import cluster
import cv2

input_path = "/Users/Srikanth/PycharmProjects/DataMiningClass/datasets/testSample_copy.jpg"

img = cv2.imread(input_path)

kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(img.reshape(-1, 3))

means = kmeans.cluster_centers_



