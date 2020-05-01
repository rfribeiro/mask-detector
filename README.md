# Project Mask Usage predictor

## Objective:

The Covid-19 changed our lives and has changed our habits and many of us entered in social isolation.

There are citys and places like marts that it is an obligation to use mask when you are on the streets or inside it.

Thinking on this my idea is to create a sofware to predict is a person is using mask or not.

The objective of this project is to train a deep convolutional neural network on images collected from internet to check if a person is using mask or not.

I used as reference the blog post ["Fine-tuning with Keras and Deep Learning"](https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/) from the guru of computer vision Adrian Rosebrock to train a new network using transfer learning process.

Finally I use a two step system to first **"detect"** faces on an image using OpenCV and after pass the found faces in this **"mask predictor"** that returns if the face is using a mask or not.

## Preprocess

In order to train a neural network we need a lot of images of our target classes and no other place to find images than iternet.

The file **"01-preprocess.ipynb"** is responsible for this.

I write a crawler using [icrawler](https://pypi.org/project/icrawler/) to search some specific phrases on the internet using Bing and download the images from people using and without using mask.

The second step is to find the faces on the images and label them. For this process I did it in three steps.
- Automatically find faces using OpenCV face detector and check visually.
- Select faces manually using Region of Interest from OpenCV.
- Use LabelImg to manually do the annotation of bounding box containing images (the idea is to use them after to train a SSD detector)

I did some hard job here manually annotating faces on the images.

**The final dataset has 4297 mask and 4419 no-mask images.**

But as in all deep learning project the more images the better and if anyone like to help it will be very welcome. please contact me.

**I did not provide the images because I collect them from internet and did not had time to check the copyright of all of them.**

## Train

The neural network used is a [MobileNet V2](https://keras.io/applications/) with the fine tuning modification on top of it.

I used **Keras** library to create the network.

The images were split on training and validation sets on 80/20 proportion. The numbers are 6974 for training and 1742 for validation.

The images are on a folder called **"data/train"** be aware of this.

The file that did the training is the **02-train_mask_classifier.ipynb**

The network was trained dusring 50 epochs and the final accuracy of around **90%** on the validation samples.

IMAGE OF CONFUSION MATRIX

## Run

The file **03-detect_face_using_mask.ipynb** that did the inferences to detect the faces on a image and predict if it is using mask or not.

You can run the prediction from an image selected from your cpu or using the webcam. Please change the **** variable to choose between them.

OR

Run the **03-detect_face_using_mask.py** using **python 03-detect_face_using_mask.py** passing an image on the command line **--image** or the default will launch the webcam. Pleas einstall the requirements before using **conda install --yes --file requirements.txt**

```
conda install --yes --file requirements.txt

python 03-detect_face_using_mask.py
```

The models are in a folder called **"model"** and pre-trained model are on drive folder [MODELS HERE!!](https://drive.google.com/open?id=1WNttcVDXo49R9hTG3P3J5iHB73dNTf2B)
- res10_300x300_ssd_iter_140000.caffemodel
- deploy.prototxt.txt
- mask_mobile_net.h5

## Improvements and drawnbacks

- Label more images
- Include no face images class to train when the detect faces send bad images
- Improve accuracy performance
- Use a better or train a new face detector (because of mask the performance is not so good)
- Create a Android application
- Create a Web application

## References
- Images from Internet

- [Fine-tuning with Keras and Deep Learning by Adrian from PyImageSearch](https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/)

- [WIDER FACE Dataset](http://shuoyang1213.me/WIDERFACE/)

- [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

- [Using Webcam on Jupyter notebooks from Konstantin Taletskiy](https://github.com/ktaletsk/NCCV/blob/master/1_OpenCV_Jupyter_webcam.ipynb)