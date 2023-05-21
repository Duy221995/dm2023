import numpy as np  # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import math

mode = []


def distance(x, xi):
    return np.sqrt(np.sum((x - xi) ** 2))


def gaussian_kernel(distance, bandwidth):
    val = (1 / (bandwidth * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((distance / bandwidth)) ** 2)
    return val


def shiftMode(point, datas, h):
    weights = []
    weighted_sum = 0
    for i in datas:
        weights.append(gaussian_kernel(distance(point, i), h))
    for i in range(len(datas)):
        weighted_sum += sum([datas[i] * weights[i]])

    if sum(weights) == 0:
        return point
    else:
        return weighted_sum / sum(weights)


def loadImage(image):
    image = plt.imread(image)
    m, n = image.shape[0:2]
    num_points = m * n
    feature_vector = np.reshape(image, (num_points, 3)).T
    return feature_vector

for i in range(len(loadImage("input_image.jpg"))):
    mode.append([])


def meanShift(h, r):
    datas = loadImage("input_image.jpg")

    for i in range(len(datas)):
        m = 0
        mode[i].append(datas[i])
        while True:
            new_point = shiftMode(mode[i][m], datas, h)
            mode[i].append(new_point)
            m += 1
            if abs(mode[i][m] - mode[i][m - 1]) < r:
                break
            mode[i][0] = mode[i][m]

    clusters = []
    for i in range(len(datas)):
        if mode[i][0] not in clusters:
            clusters.append(mode[i][0])

    return clusters

segmented_image = meanShift(5,5)
plt.figure('segmented image')
plt.set_cmap('gray')
plt.imshow(segmented_image)