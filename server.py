from flask import Flask, render_template, Response,redirect
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from PIL import Image as im
from sklearn.preprocessing import LabelBinarizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


path = "sign_mnist_cnn_10_Epochs.h5"
model = tf.keras.models.load_model(path)

labelbinarizer = LabelBinarizer()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def getLetters(result):
    classLabels = {0: 'A',
                    1: 'B',
                    2: 'C',
                    3: 'D',
                    4: 'E',
                    5: 'F',
                    6: 'G',
                    7: 'H',
                    8: 'I',
                    9: 'J',
                    10: 'K',
                    11: 'L',
                    12: 'M',
                    13: 'N',
                    14: 'O',
                    15: 'P',
                    16: 'Q',
                    17: 'R',
                    18: 'S',
                    19: 'T',
                    20: 'U',
                    21: 'V',
                    22: 'W',
                    23: 'X',
                    24: 'Y',
                    25: 'Z'}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "Error"

def check_left_hand(cap):
    if not cap.isOpened():
      print ("Could not open cam")
      exit()
    while True:
        ret , frame = cap.read()
        frame = cv2.flip(frame , 1)
        roi = frame[100:400, 30:320]
        cv2.imshow('roi' , roi)
        roi = cv2.cvtColor(roi  , cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi , (28 , 28) , interpolation= cv2.INTER_AREA)
        cv2.imshow('roi is printed into gray' , roi)
        new_roi = im.fromarray(roi)
        new_roi.save("roi.png")
        roi = cv2.imread("roi.png")
        roi = roi / 255
        roi = roi.reshape(1 , 28 , 28 , 3)
        copy = frame.copy()
        cv2.rectangle(copy ,(30 , 100 ), (320 , 400), (255, 0,0) , 5)
        captured_image= model.predict(roi ,1, verbose =0)
        result = labelbinarizer.inverse_transform(np.array(captured_image.round() , dtype = np.int32))
        cv2.putText(copy , getLetters(result) , (300 , 100) , cv2.FONT_HERSHEY_COMPLEX , 2 , (0,255,0) , 2)
        cv2.imshow('frame' , copy)
        if cv2.waitKey(1) == 13:
            break

def check_right_hand(cap):
    if not cap.isOpened():
        print("Could not open cam")
        exit()
    while True:
        ret , frame = cap.read()
        frame = cv2.flip(frame , 1)
        roi = frame[100:400 , 320:620]
        cv2.imshow('roi' , roi)
        roi = cv2.cvtColor(roi , cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi , (28 , 28) , interpolation = cv2.INTER_AREA)
        cv2.imshow('roi is printed into gray' , roi)
        new_roi = cv2.flip(roi  , 1)
        new_roi = im.fromarray(new_roi)
        new_roi.save("roi.png")
        roi = cv2.imread("roi.png")
        roi = roi / 255
        roi = roi.reshape(1 , 28 , 28 , 3)
        copy = frame.copy()
        cv2.rectangle(copy , (320 , 100) ,(620 , 400) ,  (255, 0,0) , 5)
        captured_image = model.predict(roi , 1 , verbose =0)
        result = labelbinarizer.inverse_transform(np.array(captured_image.round() , dtype = np.int32))
        cv2.putText(copy , getLetters(result) , (300 , 100) , cv2.FONT_HERSHEY_COMPLEX , 2 , (0,255,0) , 2)
        cv2.imshow('frame' , copy)
        if cv2.waitKey(1) == 13:
            
            break

TAY = 0
cap = cv2.VideoCapture(0)
if TAY == 0:#CHECK LEFT HAND
    check_left_hand(cap)
else:#CHECK RIGHT HAND
    check_right_hand(cap)
cap.release()

cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)