import math
import os
import time
from typing import Dict

import numpy as np
import tensorflow as tf
import keras
import argparse

from PIL import Image
import cv2
from keras import Model

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections',
                'detection_boxes',
                'detection_scores',
                'detection_classes'
            ]:
                tensor_name = key + ':0'

                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            return output_dict


def load_detection_graph(path_to_checkpoint):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def build_map(path_to_labelmap):
    int2category = {}
    lines = open(path_to_labelmap, 'r').read().splitlines()

    for line in lines:
        integer, category = line.split()
        int2category[int(integer)] = category

    return int2category


def get_sub_image_for_position_classification(image: np.ndarray, x1: float, y1: float, x2: float, y2: float,
                                              fixed_width: int, fixed_height: int) -> np.ndarray:
    image_width, image_height = image.shape[1], image.shape[0]

    left, right, top, bottom = x1, x2, y1, y2
    center_x = left + (right - left) / 2
    center_y = top + (bottom - top) / 2

    new_left = max(0, int(center_x - fixed_width / 2))
    new_right = min(image_width, int(center_x + fixed_width / 2))
    new_top = max(0, int(center_y - fixed_height / 2))
    new_bottom = min(image_height, int(center_y + fixed_height / 2))

    pad_left = abs(min(0, int(center_x - fixed_width / 2)))
    pad_right = abs(min(0, image_width - int(center_x + fixed_width / 2)))
    pad_top = abs(min(0, int(center_y - fixed_height / 2)))
    pad_bottom = abs(min(0, image_height - int(center_y + fixed_height / 2)))

    sub_image = image[new_top:new_bottom, new_left:new_right]

    # Due to rounding errors, we might need to add an extra pixel here to the padding to reach the desired width/height
    while (new_right - new_left + pad_left + pad_right) < fixed_width:
        pad_right += 1
    while (new_bottom - new_top + pad_top + pad_bottom) < fixed_height:
        pad_bottom += 1

    padded_sub_image = np.stack(
        [np.pad(sub_image[:, :, c], [(pad_top, pad_bottom), (pad_left, pad_right)], mode='symmetric') for c in
         range(3)], axis=2)

    return padded_sub_image


def predict_position_classification(sub_image: np.ndarray,
                                    position_classification_graph: Model,
                                    mapping: Dict[int, str]):
    class_predictions = position_classification_graph.predict(np.expand_dims(sub_image, axis=0))
    most_likely_class = np.argmax(class_predictions, axis=1)
    prediction = mapping[int(most_likely_class)]
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performs detection over input image given a trained detector.')
    parser.add_argument('--detection_inference_graph', type=str, default="2018-03-09_faster_rcnn_mensural_detection.pb",
                        help='Path to the frozen inference graph.')
    parser.add_argument('--classification_inference_graph', type=str,
                        default="2018-03-26_inception_resnet_v2_position_classification.h5",
                        help='Path to the frozen inference graph.')
    parser.add_argument('--input_image', type=str, default="12673.JPG", help='Path to the input image.')
    parser.add_argument('--detection_label_map', type=str, default="category_mapping.txt",
                        help='Path to the label map, which maps each category name to a unique number.'
                             'Must be a simple text-file with one mapping per line in the form of:'
                             '"<number> <label>", e.g. "1 barline".')
    parser.add_argument('--classification_label_map', type=str, default="position_mapping.txt",
                        help='Path to the label map, which maps each category name to a unique number.'
                             'Must be a simple text-file with one mapping per line in the form of:'
                             '"<number> <label>", e.g. "1 L1".')
    parser.add_argument('--ignorable_classes_list', type=str, default="ignorable_classes.txt",
                        help='Path to the list of classes, that should be ignored. One class per line')
    parser.add_argument('--output_image', type=str, default="annotated_image.jpg",
                        help='Path to the output image, with highlighted boxes.')
    parser.add_argument('--output_result', type=str, default="output_transcript.txt",
                        help='Path to the output file, that will contain a list of detection, '
                             'including position-classification')
    args = parser.parse_args()

    # Uncomment the next line on Windows to run the evaluation on the CPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Build category map
    detection_category_mapping = build_map(args.detection_label_map)
    classification_category_mapping = build_map(args.classification_label_map)
    ignorable_classes = []
    if args.ignorable_classes_list is not None:
        with open(args.ignorable_classes_list, "r") as ignorable_classes_list:
            ignorable_classes = ignorable_classes_list.read().splitlines()

    # Read frozen graphs
    start_time = time.time()
    detection_graph = load_detection_graph(args.detection_inference_graph)
    position_classification_graph = keras.models.load_model(args.classification_inference_graph)
    start_time_after_loading = time.time()
    fixed_height_for_position_classification = int(position_classification_graph.input.shape[1])
    fixed_width_for_position_classification = int(position_classification_graph.input.shape[2])

    # PIL Image
    image = Image.open(args.input_image)
    (image_width, image_height) = image.size

    # Opencv Image (draw)
    image_cv = cv2.imread(args.input_image, True)

    # Numpy image
    image_np = np.array(image.getdata()).reshape((image_height, image_width, 3)).astype(np.uint8)

    # Actual detection;
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    output_lines = []

    for idx in range(output_dict['num_detections']):
        if output_dict['detection_scores'][idx] > 0.5:

            y1, x1, y2, x2 = output_dict['detection_boxes'][idx]

            y1 = y1 * image_height
            y2 = y2 * image_height
            x1 = x1 * image_width
            x2 = x2 * image_width
            detected_class = detection_category_mapping[output_dict['detection_classes'][idx]]

            sub_image = get_sub_image_for_position_classification(image_np, x1, y1, x2, y2,
                                                                  fixed_width_for_position_classification,
                                                                  fixed_height_for_position_classification)

            if detected_class in ignorable_classes:
                # TODO: How to handle this case? Is an empty classification acceptable? I think it is better than a fixed (and potentially misleading or incorrect classification, such as S3).
                position_classification = ""
            else:
                position_classification = predict_position_classification(sub_image,
                                                                          position_classification_graph,
                                                                          classification_category_mapping)

            output_line = "{0:.3f},{1:.3f},{2:.3f},{3:.3f};{4};{5}".format(x1, y1, x2, y2, detected_class,
                                                                           position_classification)
            print(output_line)
            output_lines.append(output_line)

            if args.output_image is not None:
                cv2.rectangle(image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                cv2.putText(image_cv, detected_class + "/" + position_classification, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                            cv2.LINE_AA)

        else:
            break

    if args.output_image is not None:
        cv2.imwrite(args.output_image, image_cv)

    with open(args.output_result, "w") as output_file:
        output_file.write("\n".join(output_lines))

    end_time = time.time()
    print("Full execution time: {0} seconds".format(end_time - start_time))
    print("Inference time: {0} seconds".format(end_time - start_time_after_loading))
