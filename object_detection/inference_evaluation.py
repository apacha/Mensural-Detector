import argparse
import pickle
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from PIL import Image
from numba import float32

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def bb_intersection_over_union(boxA, boxB):
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


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performs detection over input image given a trained detector.')
    parser.add_argument('--inference_graph', dest='inference_graph', type=str, required=True,
                        help='Path to the frozen inference graph.')
    parser.add_argument('--label_map', dest='label_map', type=str, required=True,
                        help='Path to the label map, which is json-file that maps each category name to a unique number.',
                        default="mapping.txt")
    parser.add_argument('--number_of_classes', dest='number_of_classes', type=int, default=32,
                        help='Number of classes.')
    parser.add_argument('--input_image', dest='input_image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output_image', dest='output_image', type=str, default='detection.jpg',
                        help='Path to the output image.')
    args = parser.parse_args()

    input_image = "object_detection/samples/12673.JPG"  # args.input_image
    output_image = "object_detection/samples/12673_detection_analysis.jpg"  # args.output_image
    annotations_path = "object_detection/samples/12673.JPG.txt"
    sample_detections = "object_detection/samples/output_dict.pickle"

    image = Image.open(input_image)

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)

    # Actual detection.
    with open(sample_detections, "rb") as pickle_file:
        output_dict = pickle.load(pickle_file)

    detection_boxes = output_dict['detection_boxes']
    detection_scores = output_dict['detection_scores']

    with open(annotations_path, 'r') as gt_file:
        lines = gt_file.read().splitlines()

    gt_boxes = []  # type: List[Tuple[float32, float32, float32, float32]]
    gt_classes = []  # type: List[str]
    for line in lines:
        upper_left, lower_right, class_name, gt_position = line.split(';')
        image_width, image_height = image.size
        xmin, ymin = upper_left.split(',')
        xmax, ymax = lower_right.split(',')
        gt_classes.append(class_name)
        gt_boxes.append((float(xmin) / image_width,
                         float(ymin) / image_height,
                         float(xmax) / image_width,
                         float(ymax) / image_height))
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)

    i = 0

    for detection_box in detection_boxes:
        for gt_box in gt_boxes:
            iou = bb_intersection_over_union(detection_box, gt_box)
            if iou > 0.9:
                i += 1

    # # Visualization of the results of a detection.
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     output_dict['detection_boxes'],
    #     output_dict['detection_classes'],
    #     output_dict['detection_scores'],
    #     category_index,
    #     instance_masks=output_dict.get('detection_masks'),
    #     use_normalized_coordinates=True,
    #     line_thickness=2)
    # Image.fromarray(image_np).save(output_image)
