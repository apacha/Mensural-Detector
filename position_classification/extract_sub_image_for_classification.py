import argparse

import os
from cv2 import cv2

from tqdm import tqdm


def get_positions_of_crops_with_additional_vertical_information(upper_left, lower_right, top_offset, bottom_offset,
                                                                image_width, image_height):
    x1, y1 = upper_left.split(',')
    x2, y2 = lower_right.split(',')
    # String to float, float to int
    x1 = int(float(x1))
    y1 = max(0, int(float(y1) - top_offset))
    x2 = int(float(x2))
    y2 = min(image_height, int(float(y2) + bottom_offset))

    return x1, y1, x2, y2


def get_positions_of_fixed_size_crops(upper_left, lower_right, fixed_width, fixed_height, image_width,
                                      image_height) -> (int, int, int, int):
    """
        Returns the coordinates of a 'fixed sized' crop with the object of interest in the center.
        Note, that the resulting positions might be smaller than fixed_width and fixed_height to fit within
        the image dimensions.
    """
    left, top = upper_left.split(',')
    right, bottom = lower_right.split(',')
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
    parser = argparse.ArgumentParser(description='Draws the bounding boxes for all image')
    parser.add_argument('-dataset_directory', dest='dataset_directory', type=str, default="../db",
                        help='Path to the folder containing all images and annotations')
    parser.add_argument('-output_directory', dest='output_directory', type=str, default="data",
                        help='Path to the folder where the images should be placed, containing one sub-folder per class')
    args = parser.parse_args()

    dataset_directory = args.dataset_directory
    output_directory = args.output_directory
    files = os.listdir(dataset_directory)
    images = sorted([file for file in files if file.endswith(".JPG")])
    annotations = sorted([file for file in files if file.endswith(".txt")])
    pairs = zip(images, annotations)

    os.makedirs(output_directory, exist_ok=True)

    for pair in tqdm(pairs, desc="Extracting sub-images for each annotated symbol", total=len(images)):
        image_path = os.path.join(dataset_directory, pair[0])
        img = cv2.imread(image_path, True)
        image_height = img.shape[0]
        image_width = img.shape[1]
        annotations_path = os.path.join(dataset_directory, pair[1])

        with open(annotations_path, 'r') as gt_file:
            lines = gt_file.read().splitlines()

        top_offset = bottom_offset = 160  # We add 160px at the top and the bottom to include the staff-lines that are required to determine the position
        fixed_width, fixed_height = 128, 448
        annotation_index_inside_file = 1
        for line in lines:
            upper_left, lower_right, gt_shape, gt_position = line.split(';')

            # x1, y1, x2, y2 = get_positions_of_crops_with_additional_vertical_information(upper_left, lower_right, top_offset, bottom_offset, image_width, image_height)
            x1, y1, x2, y2 = get_positions_of_fixed_size_crops(upper_left, lower_right, fixed_width, fixed_height,
                                                               image_width, image_height)

            sub_image = img[y1:y2, x1:x2]
            filename = "{0}-{1}.png".format(pair[0], annotation_index_inside_file)
            output_filename_path = os.path.join(output_directory, gt_position, filename)
            os.makedirs(os.path.join(output_directory, gt_position), exist_ok=True)
            cv2.imwrite(output_filename_path, sub_image)
            annotation_index_inside_file += 1
