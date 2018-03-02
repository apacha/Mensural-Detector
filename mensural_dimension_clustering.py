import random
from glob import glob
from typing import Tuple

import cv2
import numpy
import pandas
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return numpy.array(similarities)


def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n


def kmeans(X: numpy.ndarray, centroids: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    N = X.shape[0]
    k, dim = centroids.shape
    prev_assignments = numpy.ones(N) * (-1)
    iter = 0

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = numpy.array(D)  # D.shape = (N,k)
        mean_IOU = numpy.mean(D)

        # assign samples to centroids
        assignments = numpy.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            return mean_IOU, centroids

        # calculate new centroids
        centroid_sums = numpy.zeros((k, dim), numpy.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j] / (numpy.sum(assignments == j))

        prev_assignments = assignments.copy()


def visualize_anchors(anchors: numpy.ndarray, visualization_width: int = 1000, visualization_height: int = 1000):
    colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 100, 0),
              (0, 255, 100), (255, 255, 255), (100, 255, 55)]

    blank_image = numpy.zeros((visualization_height, visualization_width, 3), numpy.uint8)

    stride_h = 10
    stride_w = 10

    for i in range(len(anchors)):
        (w, h) = anchors[i]
        w = int(w * visualization_width)
        h = int(h * visualization_height)
        # print(w, h)
        left_upper_corner = (10 + i * stride_w, 10 + i * stride_h)
        right_lower_corner = (left_upper_corner[0] + w, left_upper_corner[1] + h)
        cv2.rectangle(blank_image, left_upper_corner, right_lower_corner, colors[i])

    cv2.imwrite("anchors-{0}.png".format(len(anchors)), blank_image)


if __name__ == "__main__":

    annotation_dimensions = pandas.read_csv("db/bounding_box_dimensions_relative.csv")
    # visualization_width, visualization_height = 1000, 1000

    # Use this part for cropped images
    all_images = glob("db/*.JPG")
    sizes = []
    for cropped_image in tqdm(all_images, desc="Collecting image sizes"):
        image = Image.open(cropped_image)
        sizes.append(image.size)
    sizes_df = pandas.DataFrame(sizes, columns=["width", "height"])
    visualization_width, visualization_height = sizes_df["width"].mean(), sizes_df["height"].mean()
    print("Average image size: {0:.0f}x{1:.0f}px".format(visualization_width, visualization_height))
    print("Minimum image size: {0:.0f}x{1:.0f}px".format(sizes_df["width"].min(), sizes_df["height"].min()))
    print("Maximum image size: {0:.0f}x{1:.0f}px".format(sizes_df["width"].max(), sizes_df["height"].max()))

    total_number_of_clusters_to_evaluate = 10

    annotation_dimensions.plot.scatter(x='width', y='height', s=0.1, c='red')
    plt.show()

    dims = annotation_dimensions[['width', 'height']].as_matrix()

    statistics = []

    for num_clusters in tqdm(range(1, total_number_of_clusters_to_evaluate + 1), desc="Computing clusters"):
        indices = [random.randrange(dims.shape[0]) for i in range(num_clusters)]
        initial_centroids = dims[indices]
        meanIntersectionOverUnion, centroids = kmeans(dims, initial_centroids)
        statistics.append((num_clusters, meanIntersectionOverUnion, centroids))

    grid_size = 32
    with open("dimension_clustering_protocol.txt", "a") as dimension_clustering_protocol:
        for (clusters, iou, centroids) in statistics:
            overview = "{0} clusters: {1:.4f} mean IOU".format(clusters, iou)
            print(overview)
            dimension_clustering_protocol.writelines(overview + "\n")
            scales = []
            for c in centroids:
                detailed_results = "[{0:.4f} {1:.4f}] - Ratio: {2:.4f} = {3:.0f}x{4:.0f}px scaled " \
                                   "to {5:.0f}x{6:.0f} image".format(c[0], c[1], c[0] / c[1],
                                                                     c[0] * visualization_width,
                                                                     c[1] * visualization_height, visualization_width,
                                                                     visualization_height)
                print(detailed_results)
                dimension_clustering_protocol.write(detailed_results + "\n")
                scales.append(c[0] * visualization_width / grid_size)
                scales.append(c[1] * visualization_height / grid_size)
            scales.sort()
            scales = "Scales relative to {0}x{0} grid: {1}".format(grid_size, ["{0:.2f}".format(x) for x in scales])
            print(scales)
            dimension_clustering_protocol.write(scales + "\n")
            visualize_anchors(centroids, int(visualization_width), int(visualization_height))
