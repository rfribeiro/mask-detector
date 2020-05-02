# Covid-19 - Face Mask Detector using Deep Learning

![Face Mask Detector](images/my.png)
## Objective:

The Covid-19 changed our lives and has changed our habits and also because of it many of us entered social isolation.

There are cities that are an obligation to use a mask when you are on the streets and also places like marts, supermarkets, etc that you have to use masks when you are inside of them.

Thinking about this problem, my idea is to create software to automatically detect if a person is using a mask or not.

The objective of this project is to train a deep convolutional neural network on images collected from the internet to check if a person is using a mask or not.

I used as reference the blog post ["Fine-tuning with Keras and Deep Learning"](https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/) from the guru of computer vision Adrian Rosebrock to train a new network using the transfer learning process.

Finally, I use a two-step system to first **"detect"** faces on an image, using a trained face detector from OpenCV and after pass the found faces on the **"mask predictor"** that returns if the face is using a mask or not.


## Preprocessing

In order to train a neural network, we need a lot of images of our target classes and no other place to find images that the internet.

The first step is to collect images from the internet.

I write a crawler using [icrawler](https://pypi.org/project/icrawler/) to search some specific phrases on the internet using Bing and download the images from people using masks and without using it.

I also used a google extension from chrome called  [Download All Images](https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm)

It is not good to have duplicated images on the dataset, as we have a lot of images to check and look at each one to see the duplicates. I searched for an automatic tool or script to remove duplicates images from the downloaded data. And as always, I found another excellent one on Adrian's blog that works like a charm with a small change "[Detect and remove duplicate images from a dataset for deep learning by Adrian from PyImageSearch](https://www.pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/)"

The second step is to find the faces on the images and label them.

For this process, I did it in three steps.
- Automatically find faces using OpenCV face detector and check visually.
- Select faces manually using Region of Interest from OpenCV.
- Use LabelImg to manually do the annotation of bounding box containing images (the idea is to use them after to train an SSD detector)

I did some hard job here manually annotating faces on the images.

The final dataset has 4297 masks and 4419 no-mask images for training and more images, around 1000 of each class for tests.

But as in all deep learning project the more images the better and if anyone like to help it will be very welcome. please contact me.

I did not provide the dataset of images because I collect them scraping from the internet and I did not have time to check the copyright of all of them.

The file **"01-preprocess.ipynb"** is responsible for this.


**The final dataset has 4297 mask and 4419 no-mask images.**

But as in all deep learning project the more images the better and if anyone like to help it will be very welcome. please contact me.

**I did not provide the images because I collect them from internet and did not had time to check the copyright of all of them.**

## Training

The neural network used is a [MobileNet V2](https://keras.io/applications/) with the fine-tuning modification on top of it. I used the **Keras** library to create the network.

```python
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

base_model = MobileNetV2(weights='imagenet', include_top=False,input_shape=None, input_tensor=input_tensor)

headModel = base_model.output
headModel = AveragePooling2D(pool_size=(5, 5))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(N_CLASSES, activation="softmax")(headModel)
```

The images were split into training and validation sets on 80/20 proportion. The numbers are around 6974 for training and 1742 for validation. 

I used the **ImageGenerator** from Keras to automatically split the dataset. I did some modifications to augment the images, for that I used the [ImgAug](https://github.com/aleju/imgaug) library and select many of image transformation on the data.

``` python
train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=25,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2) # set validation split
```

The images are on a folder called **"data/train"** be aware of this.

The file that did the training is the **02-train_mask_classifier.ipynb**

The network was trained dusring 50 epochs and the final accuracy of around **90%** on the validation samples.

## Run

The file **03-detect_face_using_mask.ipynb** that did the inferences to detect the faces on a image and predict if it is using mask or not.

You can run the prediction from an image selected from your cpu or using the webcam.

Please change the **SOURCE** variable to choose between them.

OR

Run the **03-detect_face_using_mask.py** using **python 03-detect_face_using_mask.py** passing an image on the command line **--image** or the default will launch the webcam. 

The script uses two steps the first one to "detect" faces on the images and it uses the OpenCV library and after, all faces found pass to the "mask" predictor to check if it is using a mask or not.

The script works with many faces on the screen/images.

Please install the requirements before running the python script.

``` shell
conda install --yes --file requirements.txt

python 03-detect_face_using_mask.py
```

![Mask Face Detector in Action](images/Mask_Detector_using_Deep_Learning.gif)

I recorded a video with the mask detector in action and posted it on YouTube https://youtu.be/HqWSC5dwYYw

The models are in a folder called **"model"** and the pre-trained models are on drive folder [MODELS HERE!!](https://drive.google.com/open?id=1WNttcVDXo49R9hTG3P3J5iHB73dNTf2B)
- res10_300x300_ssd_iter_140000.caffemodel
- deploy.prototxt.txt
- mask_mobile_net.h5

## Improvements and drawnbacks

- Label more images
- Include no face images class to train when the detect faces send bad images
- Improve accuracy performance
- Use a better or train a new face detector (because of mask the performance is not so good)
- Create an Android application
- Create a Web application

## References
- Images from the Internet

- [Fine-tuning with Keras and Deep Learning by Adrian from PyImageSearch](https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/)

- [Detect and remove duplicate images from a dataset for deep learning by Adrian from PyImageSearch](https://www.pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/)

- [Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)

- [WIDER FACE Dataset](http://shuoyang1213.me/WIDERFACE/)

- [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

- [Using Webcam on Jupyter notebooks from Konstantin Taletskiy](https://github.com/ktaletsk/NCCV/blob/master/1_OpenCV_Jupyter_webcam.ipynb)