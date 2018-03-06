import numpy as np
import cv2
import argparse


def draw_bounding_boxes_into_image(image_path: str, ground_truth_annotations_path: str, destination_path: str):
    img = cv2.imread(image_path, True)

    with open(ground_truth_annotations_path, 'r') as gt_file:
        lines = gt_file.read().splitlines()

    for index, line in enumerate(lines):
        upper_left, lower_right, gt_shape, gt_position = line.split(';')

        x1, y1 = upper_left.split(',')
        x2, y2 = lower_right.split(',')

        # String to float, float to int
        x1 = int(float(x1))
        y1 = int(float(y1))
        x2 = int(float(x2))
        y2 = int(float(y2))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.putText(img, gt_shape + '/' + gt_position + '/' + str(index + 1), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    cv2.imwrite(destination_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw the bounding boxes from the ground-truth data.')
    parser.add_argument('-img', dest='img_path', type=str, required=True, help='Path to the image.')
    parser.add_argument('-gt', dest='gt_path', type=str, required=True, help='Path to the ground truth.')
    parser.add_argument('-save', dest='save_img', type=str, required=True, help='Path to save the processed image.')
    args = parser.parse_args()

    draw_bounding_boxes_into_image(args.img_path, args.gt_path, args.save_img)
