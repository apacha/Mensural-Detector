import argparse
import os
from typing import List

import pandas
from PIL import Image
from tqdm import tqdm


def create_statistics(dataset_directory: str,
                      image_file_names: List[str],
                      annotation_file_names: List[str],
                      exported_absolute_dimensions_file_name: str,
                      exported_relative_dimensions_file_name: str):
    exported_absolute_dimensions_file_path = os.path.join(dataset_directory, exported_absolute_dimensions_file_name)
    exported_relative_dimensions_file_path = os.path.join(dataset_directory, exported_relative_dimensions_file_name)

    if os.path.exists(exported_absolute_dimensions_file_path):
        os.remove(exported_absolute_dimensions_file_path)

    if os.path.exists(exported_relative_dimensions_file_path):
        os.remove(exported_relative_dimensions_file_path)

    absolute_dimensions = []
    relative_dimensions = []
    for image_file_name, annotation_file_name in tqdm(zip(image_file_names, annotation_file_names),
                                                      desc='Parsing annotation files',
                                                      total=len(image_file_names)):
        image = Image.open(os.path.join(dataset_directory, image_file_name), "r")  # type: Image.Image
        image_width = image.width
        image_height = image.height

        with open(os.path.join(dataset_directory, annotation_file_name), 'r') as gt_file:
            lines = gt_file.read().splitlines()

        for line in lines:
            upper_left, lower_right, class_name, gt_position = line.split(';')

            x_min, y_min = upper_left.split(',')
            x_max, y_max = lower_right.split(',')

            top, left, bottom, right = float(y_min), float(x_min), float(y_max), float(x_max)

            width = right - left
            height = bottom - top
            x_center = width / 2.0 + left
            y_center = height / 2.0 + top

            absolute_dimensions.append([class_name, left, right, top, bottom, x_center, y_center, width, height])
            relative_dimensions.append([class_name, left / image_width, right / image_width,
                                        top / image_height, bottom / image_height,
                                        x_center / image_width, y_center / image_height,
                                        width / image_width, height / image_height])

    absolute_statistics = pandas.DataFrame(absolute_dimensions,
                                           columns=["class", "xmin", "xmax", "ymin", "ymax", "x_c", "y_c", "width",
                                                    "height"])
    absolute_statistics.to_csv(exported_absolute_dimensions_file_path,
                               float_format="%.6f", index=False)
    relative_statistics = pandas.DataFrame(relative_dimensions,
                                           columns=["class", "xmin", "xmax", "ymin", "ymax", "x_c", "y_c", "width",
                                                    "height"])
    relative_statistics.to_csv(exported_relative_dimensions_file_path,
                               float_format="%.6f", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generates the statistics of the bounding box sizes, needed for dimension clustering')
    parser.add_argument('-dataset_directory', dest='dataset_directory', type=str, default="db",
                        help='Path to the folder containing all images and annotations')
    args = parser.parse_args()

    dataset_directory = args.dataset_directory
    files = os.listdir(dataset_directory)
    images = sorted([file for file in files if file.endswith(".JPG")])
    annotations = sorted([file for file in files if file.endswith(".txt")])

    create_statistics(dataset_directory,
                      images,
                      annotations,
                      "bounding_box_dimensions_absolute.csv",
                      "bounding_box_dimensions_relative.csv")
