import numpy as np
import tensorflow as tf
import argparse

from PIL import Image
import cv2

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
    lines = open(path_to_labelmap,'r').read().splitlines()
    
    for line in lines:
        integer, category = line.split()
        int2category[int(integer)] = category
        
    return int2category


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performs detection over input image given a trained detector.')
    parser.add_argument('--inference_graph', dest='inference_graph', type=str, required=True, help='Path to the frozen inference graph.')
    parser.add_argument('--input_image', dest='input_image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--label_map', dest='label_map', type=str, required=True, help='Path to the label map, which maps each category name to a unique number.')    
    parser.add_argument('--output_image', dest='output_image', type=str, default=None, help='Path to the output image, with highlighted boxes.')
    args = parser.parse_args()

    # Build category map
    int2category = build_map(args.label_map)
    
    # Read frozen graph
    detection_graph = load_detection_graph(args.inference_graph)

    # PIL Image
    image = Image.open(args.input_image)
    (im_width, im_height) = image.size
    
    # Opencv Image (draw)
    image_cv = cv2.imread(args.input_image,True)

    # Numpy image
    image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    # Actual detection;
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    
    for idx in range(output_dict['num_detections']):
        if output_dict['detection_scores'][idx] > 0.5:
        
            y1,x1, y2, x2 = output_dict['detection_boxes'][idx]
            
            y1 = y1 * image_cv.shape[0]
            y2 = y2 * image_cv.shape[0]
            x1 = x1 * image_cv.shape[1]
            x2 = x2 * image_cv.shape[1]
            category = int2category[output_dict['detection_classes'][idx]]
            
            # TODO Crop the box an run the position classifier
            
            print(str(x1)+','+str(y1)+';'+str(x2)+','+str(y2)+';'+str(category)+';'+str('L3'))
            
            if args.output_image is not None:
                cv2.rectangle(image_cv,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),3)
                cv2.putText(image_cv,category,(int(x1),int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)                
        
        else:
            break
        
        
    if args.output_image is not None:
        cv2.imwrite(args.output_image,image_cv)
