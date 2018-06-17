# Mensural Detector

This is the repository for the fast and reliable Mensural Music Symbol detector with Deep Learning, based on the Tensorflow Object Detection API and the [Music Object Detector](https://github.com/apacha/MusicObjectDetector-TF) 

![images](images/12073-detected.png)
 
The detailed results for various combinations of object-detector, feature-extractor, etc. can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/1lGHarxpoN_VkhEh_nIgnrR3Wp2-qEQxUfjKVWpUxS50/edit?usp=sharing).


# Preparing the application
This repository contains several scripts that can be used independently of each other. 
Before running them, make sure that you have the necessary requirements installed. 

## Install required libraries

- Python 3.6
- Tensorflow 1.8.0 (or optionally tensorflow-gpu 1.8.0)
- pycocotools (more [infos](https://github.com/matterport/Mask_RCNN/issues/6#issuecomment-341503509))
    - On Linux, run `pip install git+https://github.com/waleedka/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI`
    - On Windows, run `pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI`
- Some libraries, as specified in [requirements.txt](MusicObjectDetector/requirements.txt)

## Build Protobuf files on Linux

```commandline
cd research
protoc object_detection/protos/*.proto --python_out=.
```

## Build Protobuf files on Windows

> Run [`DownloadAndBuildProtocolBuffers.ps1`](MusicObjectDetector/DownloadAndBuildProtocolBuffers.ps1) to automate this step or manually build the protobufs by first installing [protocol buffers](https://developers.google.com/protocol-buffers/docs/downloads) and then run:

```commandline
cd research
protoc object_detection/protos/*.proto --python_out=.
```

Note, that you have to use [version 3.4.0](https://github.com/google/protobuf/releases/download/v3.4.0/) because of a [bug in 3.5.0 and 3.5.1](https://github.com/google/protobuf/issues/3957)

# Prepare the dataset
For preparing the dataset and transforming it into the right format used for the training, run the following commands, or use the `PrepareDatasetsForTensorflow.ps1` convenience script. 

```bash
# From [GIT_ROOT]/object_detection
python generate_mapping.py
python annotation_generator.py
python dataset_splitter.py

python create_tensorflow_record.py --data_dir=..\training_validation_test  	--set=training 		--annotations_dir=annotations 	--output_path=..\training.record 			--label_map_path=mapping.txt
python create_tensorflow_record.py --data_dir=..\training_validation_test  	--set=validation 	--annotations_dir=annotations 	--output_path=..\validation.record 		--label_map_path=mapping.txt
```


# Running the training

## Adding source to Python path
Make sure you have all required folders appended to the [Python path](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#add-libraries-to-pythonpath)

For Linux:
```bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

For Windows (Powershell):
```powershell
$pathToGitRoot = "[GIT_ROOT]"
$pathToSourceRoot = "$($pathToGitRoot)/object_detection"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot);$($pathToGitRoot)/slim"
```

## Adapting paths
First make sure, that you have all paths set correctly in the configuration that you wish to run (paths for `fine_tune_checkpoint` and all `input_path` and `label_map_path` fields).

Executing the following command will start the training with the selected configuration 
`python train.py --logtostderr --pipeline_config_path=configurations/[configuration].config --train_dir=data/[configuration]-train`


For running the training, you need to change the paths, according to your system

- in the configuration, you want to run, e.g. `configurations/faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_reduced_classes.config`
- if you use them, in the PowerShell scripts in the `training_scripts` folder.

Run the actual training script, by using the pre-defined Powershell scripts in the `training_scripts` folder, or by directly calling

```bash
# From [GIT_ROOT]/object_detection
# Start the training
python train.py --logtostderr --pipeline_config_path="[GIT_ROOT]/object_detection/configurations/[SELECTED_CONFIG].config" --train_dir="[GIT_ROOT]/object_detection/data/checkpoints-[SELECTED_CONFIG]-train"

# Start the validation
python eval.py --logtostderr --pipeline_config_path="[GIT_ROOT]/object_detection/configurations/[SELECTED_CONFIG].config" --checkpoint_dir="[GIT_ROOT]/object_detection/data/checkpoints-[SELECTED_CONFIG]-train" --eval_dir="[GIT_ROOT]/object_detection/data/checkpoints-[SELECTED_CONFIG]-validate"
```

A few remarks: The two scripts can and should be run at the same time, to get a live evaluation during the training. The values, may be visualized by calling `tensorboard --logdir=[GIT_ROOT]/object_detection/data`.

Notice that usually Tensorflow allocates the entire memory of your graphics card for the training. In order to run both training and validation at the same time, you might have to restrict Tensorflow from doing so, by opening `train.py` and `eval.py` and uncomment the respective (prepared) lines in the main function. E.g.:

```
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
```

## Training with pre-trained weights

It is recommended that you use pre-trained weights for known networks to speed up training and improve overall results. To do so, head over to the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), download and unzip the respective trained model, e.g. `faster_rcnn_inception_resnet_v2_atrous_coco` for reproducing the best results, we obtained. The path to the unzipped files, must be specified inside of the configuration in the `train_config`-section, e.g.

```
train-config: {
  fine_tune_checkpoint: "C:/Users/Alex/Repositories/MensuralObjectDetector/object_detection/data/faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08/model.ckpt"
  from_detection_checkpoint: true
}
```

> Note that inside that folder, there is no actual file, called `model.ckpt`, but multiple files called `model.ckpt.[something]`.

## Dimension clustering

For optimizing the performance of the detector, we adopted the dimensions clustering algorithm, proposed in the [YOLO 9000 paper](https://arxiv.org/abs/1612.08242).

To perform dimension clustering on the cropped images, run the following scripts:
```bash
# From [GIT_ROOT]/dimension_clustering
python generate_mensural_statistics.py
python mensural_dimension_clustering.py
```
The first script will load all annotations and create two csv-files containing the dimensions for each annotation from all images, including their relative sizes, compared to the entire image. The second script loads those statistics and performs dimension clustering, using a k-means algorithm on the relative dimensions of annotations.

# Running inference

Once you have a trained model, you can use the following procedure to detect symbols in a new image:

## Combined detection and classification

There is a ready-to-run standalone demo that uses pre-trained models. Details are described in this demo [README](demo/README.md).

## Freeze the model

A checkpoint will typically consist of three files:

* model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001,
* model.ckpt-${CHECKPOINT_NUMBER}.index
* model.ckpt-${CHECKPOINT_NUMBER}.meta

After you've identified a candidate checkpoint to export, run the following
command:

```bash
# From [GIT_ROOT]/object_detection
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH} \
    --output_directory output_inference_graph
```

On Windows, you can run the `object_detection/freeze_model.ps1` script, after setting the appropriate paths and checkpoint number inside.

Afterwards, you should see a folder named `output_inference_graph`, which contains the `frozen_inference_graph.pb`, which will be used in the next step.

## Detect objects
Perform inference on a single image by running

```bash
# From [GIT_ROOT]/object_detection
python inference_over_image.py \
    --inference_graph ${frozen_inference_graph.pb} \
    --label_map mapping.txt \
    --number_of_classes 32 \
    --input_image ${IMAGE_TO_BE_CLASSIFIED} \
    --output_image image_with_detection.jpg
```

or for an entire directory of images by running

```bash
# From [GIT_ROOT]/object_detection
python inference_over_directory.py \
    --inference_graph ${frozen_inference_graph.pb} \ 
    --label_map mapping.txt \
    --number_of_classes 32 
    --input_directory ${DIRECTORY_TO_IMAGES} \
    --output_directory ${OUTPUT_DIRECTORY}
```


# License

Copyright (c) 2018 [Alexander Pacha](http://alexanderpacha.com), [TU Wien](https://www.ims.tuwien.ac.at/people/alexander-pacha) and Jorge Calvo-Zaragoza.