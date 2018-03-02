import argparse
import os
import random
from glob import glob
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy
import seaborn
from PIL import Image
from imblearn.over_sampling import SMOTE
from pandas import DataFrame, concat, read_csv
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


def compute_clusters(dimensions: DataFrame, grid_size: int, visualization_width: int, visualization_height: int):
    # Draw a scatterplot with hexagonal clustering
    seaborn.jointplot(x="width", y="height", data=dimensions, kind="hex")
    plt.show()

    # Draw a Kernel Density Estimation plot
    seaborn.jointplot(x="width", y="height", data=dimensions, kind="kde")
    plt.show()

    # Draw a scatterplot with different colors per class
    seaborn.lmplot(x="width", y="height", data=dimensions, hue="class", fit_reg=False, scatter_kws={"s": 2},
                   legend=False)
    plt.show()

    dims = dimensions[['width', 'height']].as_matrix()
    statistics = []
    random.seed(42)
    for num_clusters in tqdm(range(1, total_number_of_clusters_to_evaluate + 1), desc="Computing clusters"):
        indices = [random.randrange(dims.shape[0]) for i in range(num_clusters)]
        initial_centroids = dims[indices]
        meanIntersectionOverUnion, centroids = kmeans(dims, initial_centroids)
        statistics.append((num_clusters, meanIntersectionOverUnion, centroids))

    for (clusters, iou, centroids) in statistics:
        report("{0} clusters: {1:.4f} mean IOU".format(clusters, iou))
        scales = []
        for c in centroids:
            report("[{0:.4f} {1:.4f}] - Ratio: {2:.4f} = {3:.0f}x{4:.0f}px scaled " \
                   "to {5:.0f}x{6:.0f} image".format(c[0],
                                                     c[1],
                                                     c[0] / c[1],
                                                     c[0] * visualization_width,
                                                     c[1] * visualization_height,
                                                     visualization_width,
                                                     visualization_height))
            scales.append(c[0] * visualization_width / grid_size)
            scales.append(c[1] * visualization_height / grid_size)
        scales.sort()
        report("Scales relative to {0}x{0} grid: {1}".format(grid_size, ["{0:.2f}".format(x) for x in scales]))
        visualize_anchors(centroids, int(visualization_width), int(visualization_height))


def resample_dataset(dimensions: DataFrame, resampling_method='svm') -> DataFrame:
    reproducible_seed = 42

    report("Class distribution before resampling")
    class_statistics = dimensions[['class']].groupby('class').size()
    report(str(class_statistics))

    report("Resampling with SMOTE ({0})".format(resampling_method))
    # See http://contrib.scikit-learn.org/imbalanced-learn/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html for a comparison between different resampling methods
    smote = SMOTE(random_state=reproducible_seed, kind=resampling_method)
    X_resampled, y_resampled = smote.fit_sample(dimensions[["width", "height"]],
                                                dimensions["class"])
    y = DataFrame(y_resampled)
    y.columns = ['class']

    report("Class distribution after resampling")
    report(str(y.groupby('class').size()))
    resampled_annotations = concat([DataFrame(X_resampled), DataFrame(y_resampled)], axis=1)  # type: DataFrame
    resampled_annotations.columns = ["width", "height", "class"]
    return resampled_annotations


def report(text):
    with open("dimension_clustering_protocol.txt", "a") as dimension_clustering_protocol:
        print(text)
        dimension_clustering_protocol.writelines(text + "\n")


def compute_average_image_size() -> Tuple[int, int]:
    all_images = glob(os.path.join(dataset_directory, "*.JPG"))
    sizes = []
    for cropped_image in tqdm(all_images, desc="Collecting image sizes"):
        image = Image.open(cropped_image)
        sizes.append(image.size)
    sizes_df = DataFrame(sizes, columns=["width", "height"])
    visualization_width, visualization_height = int(sizes_df["width"].mean()), int(sizes_df["height"].mean())
    report("Average image size: {0:.0f}x{1:.0f}px".format(visualization_width, visualization_height))
    report("Minimum image size: {0:.0f}x{1:.0f}px".format(sizes_df["width"].min(), sizes_df["height"].min()))
    report("Maximum image size: {0:.0f}x{1:.0f}px".format(sizes_df["width"].max(), sizes_df["height"].max()))
    return visualization_width, visualization_height

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Performs dimension clustering on the mensural dataset')
    parser.add_argument('-dataset_directory', dest='dataset_directory', type=str, default="../db",
                        help='Path to the folder containing all images and annotations')
    args = parser.parse_args()
    dataset_directory = args.dataset_directory

    annotation_dimensions = read_csv(os.path.join(dataset_directory, "bounding_box_dimensions_relative.csv"))

    average_width, average_height = compute_average_image_size()
    # Either use average image sizes for visualization
    # visualization_width, visualization_height = average_width, average_height
    # or override with custom width/height - Most images will be downscaled to 1000x690 or 690x1000 respectively
    visualization_width, visualization_height = 1000, 690

    grid_size = 16
    total_number_of_clusters_to_evaluate = 10

    report("Computing clusters for original data")
    compute_clusters(annotation_dimensions, grid_size, visualization_width, visualization_height)

    report("Computing clusters for resampled data (to account for class imbalance)")
    resampled_annotations = resample_dataset(annotation_dimensions)
    compute_clusters(resampled_annotations, grid_size, visualization_width, visualization_height)
