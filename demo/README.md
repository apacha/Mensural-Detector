# Running this demo

Download the trained models from 
- [2018-03-09_faster_rcnn_mensural_detection.pb](https://owncloud.tuwien.ac.at/index.php/s/SU81Uibv1gRxi0F)
- [2018-03-26_inception_resnet_v2_position_classification.h5](https://owncloud.tuwien.ac.at/index.php/s/40lYm9qD8wIhwqt)

and put them into the demo directory.

Then run `inference_over_image.py` from inside the demo directory like this, with your own image, or try one of the images, used for training, e.g. `..\db\00518.JPG`:

```bash
python inference_over_image.py \
    --detection_inference_graph 2018-03-09_faster_rcnn_mensural_detection.pb \
    --classification_inference_graph 2018-03-26_inception_resnet_v2_position_classification.h5 \
    --input_image PATH_TO_YOUR_IMAGE.jpg \
    --detection_label_map category_mapping.txt \
    --classification_label_map position_mapping.txt \
    --ignorable_classes_list ignorable_classes.txt \
    --output_image annotated_image.jpg \
    --output_result output_transcript.txt
```

