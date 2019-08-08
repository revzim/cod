# cod (custom object detection)

##

#### [Tensorflow](https://www.tensorflow.org/) and transfer learning to train a custom model for object detection.

##

![Custom-Trained Model Learning Comparison](https://raw.githubusercontent.com/revzim/cod/master/comparison.png)

###### Andy Zimmelman 2019

##

##### Environment: ` Windows 10 PC Home Edition | FX-8320 OC TO 4.0 GHZ | NVIDIA GeForce GTX 1060 6GB `


##### Before attempting custom object detection, I checked out the tutorial for [Tensorflow Image Classification](https://www.tensorflow.org/hub/tutorials/image_retraining) and successfully tested certain classification for different images. I used the images provided within that tutorial for this project to classify certain species of flowers because why not. A simplified description of my process is outlined below.

##

#### Process:
` * CONSIDER OBJECT(S) FOR DETECTION `

` * GATHER DATA AND OBJECTS `

` * LABEL DATA AND OBJECTS WITH IMAGE LABEL SOFTWARE ` (I used [labelImg](https://github.com/tzutalin/labelImg) )

` * GENERATE LABEL MAP (label_map.pbtxt) FOR EACH OBJECT CLASS `

` * labelImg STORED AND SAVED DATA AS XML WILL NEED TO BE CONVERTED TO CSV IN ORDER TO BE CONVERTED LATER TO TFRECORDS `

` * ONCE CONVERTED, THE CSV LABELED IMAGE FILES WILL BE USED TO GENERATE TFRECORDS FOR OUR THE TRAINING MODEL `

` * GENERATE TFRECORDS FROM LABELED CSV DATA (gen_tf_records.py) `

` * TFRECORDS ARE GENERATED INTO TWO SEPARATE ENTITIES: TRAINING AND TESTING `

` * CREATE OR MODIFY AN EXISTING TRAINING CONFIG (training/training.config) FOR OUR CUSTOM OBJECT DETECTION `

` * TRAIN MODEL (train.py) (this can take hours or days depending on your data and model and/or hyperparams) `

` * ONCE SATISFIED, OR TIRED OF WAITING, AN INFERENCE GRAPH, INCLUDING ANY MODEL CHECKPOINT DATA, MUST BE CREATED IN ORDER TO TEST THE CUSTOM-TRAINED MODEL `

` * TENSORFLOW PROVIDES A SCRIPT TO GENERATE THE INFERENCE GRAPH (export_inference_graph.py), WHICH MUST BE DONE IN ORDER TO TEST OUR CUSTOM-TRAINED MODEL `

` * ONCE COMPLETED, THE GENERATED INFERENCE GRAPH CAN NOW BE USED FOR TESTING OUR CUSTOM-TRAINED MODEL `

` * TESTING USES Tensorflow's Object Detection API, I HAVE PROVIDED MY TESTING FILE WITH THIS REPO (test_cod.py) `

` * THE PROVIDED TEST FILE (test_cod.py) WILL TEST OUR NEWLY CREATED CUSTOM-TRAINED MODEL AGAINST THE PROVIDED TEST IMAGE AND OUTPUT THE RESULTS WITH BOXES AND SCORES AROUND ANY OBJECTS THE MODEL DETECTS `

` * I HAVE PROVIDED IMAGES WITHIN (testers/) AS AN EXAMPLE OF THE MODELS LEARNING OVER TIME 2001 STEPS -> 20004 STEPS `

` * OLDER INFERENCE GRAPHS ARE PROVIDED, HOWEVER, OBVIOUSLY, THEY WILL YIELD WORSE RESULTS THAN THE CURRENT (inference_graph/) DIRECTORY `

##

*** This repository should be used for learning purposes. My custom-trained model that is provided within this repository has been trained for ~20000 steps.** 

*** Note: The training process according to the config file (training/training.config) limits the training process to 200000 steps, which the authors at Tensorflow found to be "sufficient enough to train the... dataset." This amount of steps will effectively bypass the learning rate schedule, leading to a never-decaying learning rate.**

*** Any implementation or use of Tensorflow requires Tensorflow, along with other libraries and API's to be installed. Installation instructions for Tensorflow can be found [here](https://www.tensorflow.org/install).**

*** Certain files and directories within this repository were moved here for viewing purposes. Thus, linking and certain files might yield warnings/errors due to the nature of this repository and the nature of Tensorflow.**

##

* [Tensorflow: Research](https://github.com/tensorflow/models/tree/master/research/object_detection)
* [Tensorflow: Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
* [Tensorflow: Training Custom Object Detector](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)
* [labelImg](https://github.com/tzutalin/labelImg)
* [Datitran Dataset](https://github.com/datitran/raccoon_dataset)
* [Tensorflow: Model Zoo for Transfer Learning & Hubs](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
