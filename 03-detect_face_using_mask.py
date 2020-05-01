# Import the required modules
import cv2
import time
import PIL.Image
from io import BytesIO
import numpy as np
import glob
import argparse

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to our input image")
ap.add_argument("-c", "--confidence", type=float, help="confidence threshold")
args = vars(ap.parse_args())

def show_webcam():
    cam = cv2.VideoCapture(0)
   
    while cam.isOpened():
        try:
            t1 = time.time()
            ret, frame = cam.read()
           
            frame = detect_face_cnn(frame)
                
            cv2.imshow("Image", frame)

            t2 = time.time()

            s = f"""{int(1/(t2-t1))} FPS"""
            print(s)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except KeyboardInterrupt:
            print()
            cam.release()
            print ("Stream stopped")
            break

    
    cam.release()
    cv2.destroyAllWindows()

# ----

# ### Detect faces on image using OpenCV
# Face detection with OpenCV and deep learning (Adrian)
# https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

# load our serialized model from disk
caffe_model = 'deploy.prototxt.txt'
caffe_trained = 'res10_300x300_ssd_iter_140000.caffemodel'
caffe_confidence = 0.30
model_folder = './model/'
mask_model = "mask_mobile_net.h5"

if args["confidence"]:
    caffe_confidence = args["confidence"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(model_folder + caffe_model, 
                               model_folder + caffe_trained
                              )


model = load_model(model_folder + mask_model)


# Detect faces on image and call mask predictor
def detect_face_cnn(image, save = False, show = False):
    
    if image is not None:
        (h, w) = image.shape[:2]
        
        image_resized = cv2.resize(image, (300, 300))

        blob = cv2.dnn.blobFromImage(image_resized, 
                                     1.0,
                                     (300, 300), 
                                     (104.0, 
                                      177.0, 
                                      123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
           
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > caffe_confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                try:
                    img_crop =   image[startY-10:endY+10, startX-10:endX+10]

                    # predict mask or not
                    pred, pred_res = predict_mask(img_crop)
                    
                    print("Face Detection confidence:{:2f}".format(round(confidence,2)), pred)

                    label = "MASK" if pred_res == 0 else "NO-MASK"
                    color = (0,255,0) if pred_res == 0 else (0,0,255)

                    cv2.putText(image, label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                    cv2.rectangle(image, (startX, startY), (endX, endY), color)
                except:
                    print("found crop errors {}".format(round(confidence,2)))

        if show:
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return image
    else:
        print("image not found!")


# Predict if face is using mask or not
def predict_mask(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    # make predictions on the input image
    pred = model.predict(image)
    pred_res = pred.argmax(axis=1)[0]
    
    return pred, pred_res


### MAIN AREA

# ### Check image source from file or Webcam

# select image or webcam
if args["image"] is not None:
    image = cv2.imread(args["image"])
    detect_face_cnn(image, show = True)
else:
    show_webcam()




