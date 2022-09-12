from email.mime import image
from re import T
from turtle import ht
import cv2
import time
import numpy as np
import handTrackModule as htm
import os

# folder = "testImages"
# myList = os.listdir(folder)
# print(myList)
# overlayList = []
# for imPath in myList:
#     image = cv2.imread(f'{folder}/{imPath}')
#     overlayList.append(image)

# print(overlayList)

# head = overlayList[0]


cam = cv2.VideoCapture(0)

detector = htm.handDetector(detConfidence=0.85)

while True:
    
    # import and flip image
    scs, img = cam.read()
    # img[0:200, 0:1300] = head
    img = cv2.flip(img, 1)

    # hand landmarks
    img = detector.detectHands(img)
    lmList = detector.detectPos(img, drawY= False)

    # check fingers up
    fingers = detector.detectUp()
    print(fingers)

    if len(lmList) != 0:
        print(lmList)


    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break