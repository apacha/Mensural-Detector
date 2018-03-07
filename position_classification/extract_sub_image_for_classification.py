import argparse

import os
from cv2 import cv2

from tqdm import tqdm


def get_positions_of_fixed_size_crops(upper_left, lower_right, fixed_width, fixed_height, image_width,
                                      image_height) -> (int, int, int, int):
    """
        Returns the coordinates of a "fixed sized" crop with the object of interest in the center.
        Note, that the resulting positions might be smaller than fixed_width and fixed_height to fit within
        the image dimensions.
    """
    left, top = upper_left.split(",")
    right, bottom = lower_right.split(",")
    # String to float, float to int
    left, right, top, bottom = float(left), float(right), float(top), float(bottom)

    center_x = left + (right - left) / 2
    center_y = top + (bottom - top) / 2
    new_left = max(0, center_x - fixed_width / 2)
    new_right = min(image_width, center_x + fixed_width / 2)
    new_top = max(0, center_y - fixed_height / 2)
    new_bottom = min(image_height, center_y + fixed_height / 2)

    return int(new_left), int(new_top), int(new_right), int(new_bottom)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draws the bounding boxes for all image")
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--dataset_directory", dest="dataset_directory", type=str, default="../db",
                        help="Path to the folder containing all images and annotations")
    parser.add_argument("--output_directory", dest="output_directory", type=str, default="data",
                        help="Path to the folder where the images should be placed, containing one sub-folder per class")
    parser.add_argument("--group_by", dest="group_by", type=str, default="staff_position",
                        help="Determines how to group the extracted sub-images. "
                             "Can be either 'staff_position' or 'class_name'.")
    parser.add_argument("--ignore_classes_without_semantic_staff_position", dest="ignore_classes_without_semantic_staff_position",
                        action='store_true',
                        help="Whether to ignore classes such as proportio maior or barline, that do not have a "
                             "semantic staff line position.")
    args = parser.parse_args()

    dataset_directory = args.dataset_directory
    output_directory = args.output_directory
    group_by = args.group_by
    ignore_classes_without_semantic_staff_position = args.ignore_classes_without_semantic_staff_position
    if group_by not in ["staff_position", "class_name"]:
        raise Exception("Group-By parameter must be either 'staff_position' or 'class_name'")

    files = os.listdir(dataset_directory)
    images = sorted([file for file in files if file.endswith(".JPG")])
    annotations = sorted([file for file in files if file.endswith(".txt")])
    pairs = zip(images, annotations)

    os.makedirs(output_directory, exist_ok=True)

    ignorable_classes = ["proportio_maior", "proportio_minor", "barline", "double_barline", "undefined", "sharp",
                         "ligature", "fermata", "beam", "dot", "cut_time", "common_time"]

    all_classes = ["ligature", "coloured_semibrevis", "coloured_brevis", "flat", "f_clef_1", "longa", "proportio_maior",
                   "semiminima_rest", "f_clef_2", "g_clef", "semibrevis", "longa_rest", "proportio_minor", "minima",
                   "coloured_minima", "coloured_semiminima", "brevis_rest", "undefined", "brevis", "barline",
                   "cut_time", "c_clef", "fermata", "beam", "minima_rest", "semibrevis_rest", "sharp", "common_time",
                   "dot", "custos", "semiminima", "double_barline"]

    for pair in tqdm(pairs, desc="Extracting sub-images for each annotated symbol", total=len(images)):
        image_path = os.path.join(dataset_directory, pair[0])
        img = cv2.imread(image_path, True)
        image_height = img.shape[0]
        image_width = img.shape[1]
        annotations_path = os.path.join(dataset_directory, pair[1])

        with open(annotations_path, "r") as gt_file:
            lines = gt_file.read().splitlines()

        top_offset = bottom_offset = 160  # We add 160px at the top and the bottom to include the staff-lines that are required to determine the position
        fixed_width, fixed_height = 128, 448

        for index, line in enumerate(lines):
            upper_left, lower_right, class_name, staff_position = line.split(";")

            if ignore_classes_without_semantic_staff_position and class_name in ignorable_classes:
                continue

            if group_by == 'staff_position':
                classification_parameter = staff_position
            else:
                classification_parameter = class_name

            x1, y1, x2, y2 = get_positions_of_fixed_size_crops(upper_left, lower_right, fixed_width, fixed_height,
                                                               image_width, image_height)

            sub_image = img[y1:y2, x1:x2]
            filename = "{0}-{1}.png".format(pair[0], index + 1)
            output_filename_path = os.path.join(output_directory, classification_parameter, filename)
            os.makedirs(os.path.join(output_directory, classification_parameter), exist_ok=True)
            cv2.imwrite(output_filename_path, sub_image)
