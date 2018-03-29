import pickle
from glob import glob
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from PIL import Image

from object_detection.utils import ops as utils_ops, label_map_util, visualization_utils as vis_util

from object_detection.inference_over_image import load_category_index, load_image_into_numpy_array

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


def bb_intersection_over_union(boxA, boxB):
    boxes_do_not_overlap = (((boxA[0] > boxB[0] and boxA[0] > boxB[2])
                             or (boxB[0] > boxA[0] and boxB[0] > boxA[2]))
                            or
                            ((boxA[1] > boxB[1] and boxA[1] > boxB[3])
                             or (boxB[1] > boxA[1] and boxB[1] > boxA[3])))
    if boxes_do_not_overlap:
        return 0

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def visualize_evaluation(input_image, output_image, annotations, serialized_detections, path_to_labels,
                         number_of_classes):
    image = Image.open(input_image)
    image_width, image_height = image.size

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    category_index = load_category_index(path_to_labels, number_of_classes)
    category_to_index = {v["name"]: v["id"] for k, v in category_index.items()}

    # Actual detection.
    with open(serialized_detections, "rb") as pickle_file:
        output_dict = pickle.load(pickle_file)

    detection_boxes_normalized = output_dict['detection_boxes']
    detection_boxes = []
    for (top, left, bottom, right) in detection_boxes_normalized:
        detection_boxes.append((top * image_height,
                                left * image_width,
                                bottom * image_height,
                                right * image_width))
    detection_scores = output_dict['detection_scores']
    detection_classes = output_dict['detection_classes']

    with open(annotations, 'r') as gt_file:
        lines = gt_file.read().splitlines()

    gt_boxes = []  # type: List[Tuple[float, float, float, float]]
    gt_classes = []  # type: List[str]
    for line in lines:
        upper_left, lower_right, class_name, gt_position = line.split(';')
        left, top = upper_left.split(',')
        right, bottom = lower_right.split(',')
        gt_classes.append(category_to_index[class_name])
        gt_boxes.append((float(top), float(left), float(bottom), float(right)))

    score_threshold = 0.5
    iou_threshold = 0.5
    correct_detections = []
    missed_detections = []
    incorrect_detections = []
    misclassified_detections = []
    for gt_box, gt_class in zip(gt_boxes, gt_classes):
        gt_box_detected = False
        for (box, score, detected_class) in zip(detection_boxes, detection_scores, detection_classes):
            if score < score_threshold:
                continue

            incorrect_detections.append(box)
            iou = bb_intersection_over_union(gt_box, box)
            if iou > iou_threshold:
                if gt_class == detected_class:
                    correct_detections.append(box)
                else:
                    misclassified_detections.append(box)
                gt_box_detected = True

        if not gt_box_detected:
            missed_detections.append(gt_box)

    incorrect_detections = list(set(incorrect_detections) - set(correct_detections) - set(misclassified_detections))
    correct_detections = correct_detections
    missed_detections = missed_detections

    print("{0} correct detections (IoU > {1}, Score > {2})".format(
        len(correct_detections), iou_threshold, score_threshold))
    print("{0} incorrect detections".format(len(incorrect_detections)))
    print("{0} missed detections".format(len(missed_detections)))
    print("{0} misclassified detections".format(len(misclassified_detections)))

    # Visualization of the results of a detection.
    line_thickness = 5
    cloned_image = image_np.copy()
    vis_util.visualize_boxes_and_labels_on_image_array(cloned_image, np.asarray(correct_detections, dtype=np.float32),
                                                       None, None, category_index, instance_masks=None,
                                                       use_normalized_coordinates=False,
                                                       line_thickness=line_thickness,
                                                       groundtruth_box_visualization_color=(0, 128, 0))
    vis_util.visualize_boxes_and_labels_on_image_array(cloned_image, np.asarray(missed_detections, dtype=np.float32),
                                                       None, None, category_index, instance_masks=None,
                                                       use_normalized_coordinates=False,
                                                       line_thickness=line_thickness,
                                                       groundtruth_box_visualization_color=(245, 245, 245))
    vis_util.visualize_boxes_and_labels_on_image_array(cloned_image, np.asarray(incorrect_detections, dtype=np.float32),
                                                       None, None, category_index, instance_masks=None,
                                                       use_normalized_coordinates=False,
                                                       line_thickness=line_thickness,
                                                       groundtruth_box_visualization_color=(255, 106, 0))
    vis_util.visualize_boxes_and_labels_on_image_array(cloned_image,
                                                       np.asarray(misclassified_detections, dtype=np.float32),
                                                       None, None, category_index, instance_masks=None,
                                                       use_normalized_coordinates=False,
                                                       line_thickness=line_thickness,
                                                       groundtruth_box_visualization_color=(255, 185, 0))
    Image.fromarray(cloned_image).save(output_image)


if __name__ == "__main__":
    input_image = "object_detection/samples/12673.JPG"  # args.input_image
    output_image = "object_detection/samples/12673_detection_analysis.jpg"  # args.output_image
    annotations = "object_detection/samples/12673.JPG.txt"
    serialized_detections = "object_detection/samples/output_dict.pickle"
    path_to_labels = "object_detection/mapping.txt"
    number_of_classes = 32

    visualize_evaluation(input_image, output_image, annotations, serialized_detections, path_to_labels,
                         number_of_classes)


    annotations = glob("db/*.txt")
    input_images = glob("db/*.JPG")
