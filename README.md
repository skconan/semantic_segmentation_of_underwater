# Semantic Segmentation of Underwater For Robosub 2019

  * This project, We apply deep learning to underwater images dataset for classify class of each pixel of an image that called "Semantic Segmentaion". 
  
  * The dataset consist of the underwater image (3 Channels) from Chulabhorn Walailak Swimming Pool at Kasetsart University and Robosub competition at San Diego, CA, USA. 
  
  * Ours model based on conventional autoencoder that have 2 important part. 
  
    - First, the encoder part, we apply Conv2D > BatchNorm > ReLU > Maxpooling (downsampling) and apply Dropout in 2 last layer of encoder part.
    
    - Second, the decoder part, we apply Conv2d > UpSampling to every layer in decoder part.
   
  * Finally, we run model on Jetson TX2 with ROS Framework and publish the result to node that have 6.5 frame per second. If you want to see the summary of model [click here](https://raw.githubusercontent.com/skconan/semantic_segmentation_of_underwater/master/Screenshot%20from%202019-07-18%2022-39-28.png).
   


## Table of Contents
**[Hardware](#hardware)**<br>
**[Software](#software)**<br>
**[Libraries](#libraries)**<br>
**[Files](#files)**<br>
**[Website](#website)**<br>

## Hardware

* uEye Industrial Cameras [**UI-3260CP Rev.2**](https://en.ids-imaging.com/store/ui-3260cp-rev-2.html)
* Kowa C-Mount 6mm
* Jetson TX2

## Software

* Robot Operating System [**ROS**](http://www.ros.org) 

## Libraries 

* Tensorflow
* Keras
* OpenCV
* Numpy
* Matplot

## Files 

* [model.py](https://github.com/skconan/semantic_segmentation_of_underwater/blob/master/source/model.py) - create structure of model.

* [mycallback.py](https://github.com/skconan/semantic_segmentation_of_underwater/blob/master/source/mycallback.py) - create callback for handle the model saving and save image while training.

* [train.py](https://github.com/skconan/semantic_segmentation_of_underwater/blob/master/source/train.py) - read image file and divide the training and validation set.

## Website

* [robin-gpu](https://robin-gpu.cpe.ku.ac.th:8000/) - This website use for sampling images from video file and label the groundTruth images.

* [Github robin-gpu](https://github.com/skconan/robin_cv_web)

## Reference

[neural network](https://medium.com/@sanparithmarukatat/สนุกกับ-neural-network-657fa293c4d1)
[A Comprehensive Introduction to Different Types of Convolutions in Deep Learning](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)


## Contact

* email: supakit.kr@gmail.com
* [Linkedin](https://www.linkedin.com/in/skconan/)


