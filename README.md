# Mensural Detector

This is the repository for the fast and reliable Mensural Music Symbol detector with Deep Learning, based on the Tensorflow Object Detection API: 
 
The detailed results for various combinations of object-detector, feature-extractor, etc. can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/1lGHarxpoN_VkhEh_nIgnrR3Wp2-qEQxUfjKVWpUxS50/edit?usp=sharing).


# Running the application
This repository contains several scripts that can be used independently of each other. 
Before running them, make sure that you have the necessary requirements installed. 

## Requirements

- Python 3.6
- Tensorflow 1.5.0 (or optionally tensorflow-gpu 1.5.0)

For installing Tensorflow, we recommend using [Anaconda](https://www.continuum.io/downloads) or 
[Miniconda](https://conda.io/miniconda.html) as Python distribution (we did so for preparing Travis-CI and it worked).

To accelerate training even further, you can make use of your GPU, by installing tensorflow-gpu instead of tensorflow
via pip (note that you can only have one of them) and the required Nvidia drivers. For Windows, we recommend the
[excellent tutorial by Phil Ferriere](https://github.com/philferriere/dlwin). For Linux, we recommend using the
 official tutorials by [Tensorflow](https://www.tensorflow.org/install/) and [Keras](https://keras.io/#installation).

### Prepare the library

First, make sure you have [protocol buffers](https://developers.google.com/protocol-buffers/docs/downloads) installed, by heading over to [the download page](https://github.com/google/protobuf/releases/download/v3.4.0/), download and install it, so you can run it in the next step.

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

### Append library to python path
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


# License

Copyright (c) 2018 [Alexander Pacha](http://alexanderpacha.com), [TU Wien](https://www.ims.tuwien.ac.at/people/alexander-pacha) and Jorge Calvo-Zaragoza