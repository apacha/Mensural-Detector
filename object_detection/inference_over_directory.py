import os
import tensorflow as tf
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
import pickle

import object_detection.inference_over_image as inference_over_image
from object_detection.utils import ops as utils_ops, label_map_util, visualization_utils as vis_util

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


def run_inference_for_single_image(image, sess, tensor_dict):
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performs detection over input image given a trained detector.')
    parser.add_argument('--inference_graph', dest='inference_graph', type=str, required=True,
                        help='Path to the frozen inference graph.')
    parser.add_argument('--label_map', dest='label_map', type=str, required=True,
                        help='Path to the label map, which is json-file that maps each category name to a unique number.',
                        default="mapping.txt")
    parser.add_argument('--input_directory', dest='input_directory', type=str, required=True,
                        help='Path to the directory that contains the images for which object detection should be performed')
    parser.add_argument('--number_of_classes', dest='number_of_classes', type=int, default=32,
                        help='Number of classes.')
    parser.add_argument('--output_directory', dest='output_directory', type=str, default='detection_output',
                        help='Path to the output directory, that will contain the results.')
    args = parser.parse_args()

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    path_to_frozen_inference_graph = args.inference_graph
    path_to_labels = args.label_map
    number_of_classes = args.number_of_classes
    input_image_directory = args.input_directory
    output_directory = args.output_directory

    # Read frozen graph
    detection_graph = inference_over_image.load_detection_graph(path_to_frozen_inference_graph)
    category_index = inference_over_image.load_category_index(path_to_labels, number_of_classes)

    input_files = os.listdir(input_image_directory)
    os.makedirs(output_directory, exist_ok=True)

    with detection_graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'

                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)


            for input_file in tqdm(input_files, desc="Detecting objects"):
                try:
                    image = Image.open(os.path.join(input_image_directory, input_file))
                except:
                    # print("Can not read {0} as image. Skipping file".format(input_file))
                    continue

                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = inference_over_image.load_image_into_numpy_array(image)

                output_dict = run_inference_for_single_image(image_np, sess, tensor_dict)

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=2)

                input_file_name, extension = os.path.splitext(os.path.basename(input_file))
                output_file = os.path.join(output_directory, "{0}_detection{1}".format(input_file_name, extension))
                Image.fromarray(image_np).save(output_file)

                output_pickle_file = os.path.join(output_directory, "{0}_detection.pickle".format(input_file_name))
                with open(output_pickle_file, "wb") as pickle_file:
                    pickle.dump(output_dict, pickle_file)
