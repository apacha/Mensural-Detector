# Mensural Detector

This is the repository for the fast and reliable Mensural Music Symbol detector with Deep Learning, based on the Tensorflow Object Detection API and the [Music Object Detector](https://github.com/apacha/MusicObjectDetector-TF) 
 
The detailed results for various combinations of object-detector, feature-extractor, etc. can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/1lGHarxpoN_VkhEh_nIgnrR3Wp2-qEQxUfjKVWpUxS50/edit?usp=sharing).


# Requirements

- Python 3.6
- Tensorflow 1.5.0 (or optionally tensorflow-gpu 1.5.0)

For installing Tensorflow, we recommend using [Anaconda](https://www.continuum.io/downloads) or 
[Miniconda](https://conda.io/miniconda.html) as Python distribution (we did so for preparing Travis-CI and it worked).

To accelerate training even further, you can make use of your GPU, by installing tensorflow-gpu instead of tensorflow
via pip (note that you can only have one of them) and the required Nvidia drivers. For Windows, we recommend the
[excellent tutorial by Phil Ferriere](https://github.com/philferriere/dlwin). For Linux, we recommend using the
 official tutorials by [Tensorflow](https://www.tensorflow.org/install/) and [Keras](https://keras.io/#installation).

## Prepare the library

First, make sure you have [protocol buffers](https://developers.google.com/protocol-buffers/docs/downloads) installed, by heading over to [the download page](https://github.com/google/protobuf/releases/), download and install it, so you can run it in the next step.

> For Windows: Notice that version 3.4.0 is required, because [3.5.0 does not work on Windows](https://github.com/google/protobuf/issues/3957).
 
Now build the required libraries:

```commandline
cd Mensural-Detector
protoc tensorflow_object_detection/protos/*.proto --python_out=.
cd slim
python setup.py install
cd ..
python setup.py install
```

See also https://github.com/tensorflow/models for additional information

## Append library to python path
Finally add the [source to the python path](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#add-libraries-to-pythonpath).
 
For Unix, it should be something like

``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

For Windows (in Powershell):

``` powershell
$pathToGitRoot = "C:/[YourPathTo]/[Mensural-Detector]"
$pathToSourceRoot = "$($pathToGitRoot)/object_detection"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot);$($pathToGitRoot)/slim"
```

Inside the PyCharm, make sure that the project structure is correctly set up and both `object_detection` and `slim` are marked as source folders. 

![PyCharm Settings](images/PyCharm%20Settings.png)

# Prepare the dataset
For preparing the dataset and transforming it into the right format used for the training, run the following commands, or use the `PrepareDatasetsForTensorflow.ps1` convenience script. 

```commandline
python generate_mapping.py
python annotation_generator.py
python dataset_splitter.py

python create_tensorflow_record.py --data_dir=..\training_validation_test  	--set=training 		--annotations_dir=annotations 	--output_path=..\training.record 			--label_map_path=mapping.txt
python create_tensorflow_record.py --data_dir=..\training_validation_test  	--set=validation 	--annotations_dir=annotations 	--output_path=..\validation.record 		--label_map_path=mapping.txt
```


# Training and Evaluation

First make sure, that you have all paths set correctly in the configuration that you wish to run (paths for `fine_tune_checkpoint` and all `input_path` and `label_map_path` fields).

Executing the following command will start the training with the selected configuration 
`python train.py --logtostderr --pipeline_config_path=configurations/[configuration].config --train_dir=data/[configuration]-train`


For running the training, you need to change the paths, according to your system

- in the configuration, you want to run, e.g. `configurations/faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_reduced_classes.config`
- if you use them, in the PowerShell scripts in the `training_scripts` folder.

Run the actual training script, by using the pre-defined Powershell scripts in the `training_scripts` folder, or by directly calling

```
# Start the training
python [GIT_ROOT]/object_detection/train.py --logtostderr --pipeline_config_path="[GIT_ROOT]/object_detection/configurations/[SELECTED_CONFIG].config" --train_dir="[GIT_ROOT]/object_detection/data/checkpoints-[SELECTED_CONFIG]-train"

# Start the validation
python [GIT_ROOT]/object_detection/eval.py --logtostderr --pipeline_config_path="[GIT_ROOT]/object_detection/configurations/[SELECTED_CONFIG].config" --checkpoint_dir="[GIT_ROOT]/object_detection/data/checkpoints-[SELECTED_CONFIG]-train" --eval_dir="[GIT_ROOT]/object_detection/data/checkpoints-[SELECTED_CONFIG]-validate"
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


# Running inference

TBD


# License

Copyright (c) 2018 [Alexander Pacha](http://alexanderpacha.com), [TU Wien](https://www.ims.tuwien.ac.at/people/alexander-pacha) and Jorge Calvo-Zaragoza.