import cv2
import numpy as np
import time
import os
import handTrackModule as htm

#######################
brushThickness = 15
eraserThickness = 100
########################


# folderPath = "Header"
# myList = os.listdir(folderPath)
# print(myList)
# overlayList = []
# for imPath in myList:
#     image = cv2.imread(f'{folderPath}/{imPath}')
#     overlayList.append(image)
# print(len(overlayList))
# header = overlayList[0]

path = './logo.jpeg'
overlayImg = cv2.imread(path)

drawColor = (255, 0, 255)

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector()
xp, yp = 0, 0
imgCanvas = np.zeros((480, 800, 3), np.uint8)

while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img[0:715, 0:1213] = overlayImg

    # 2. Find Hand Landmarks
    img = detector.detectHands(img)
    lmList = detector.detectPos(img, drawY=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        
        fingers = detector.detectUp()
        
        if fingers[1] and fingers[2] and fingers[3]:
            xp, yp = 0, 0
            drawColor = (0,0,0)
            print('eraser mode')
            cv2.rectangle(img, (x1, y1 - 35), (x2, y2 + 35), drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            # print("Selection Mode")
            
            # if y1 < 120:
            #     if 250 < x1 < 450:
            #         drawColor = (255, 0, 255)
            #     elif 780 < x1 < 1000:
            #         drawColor = (0, 0, 0)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)


        # 5. If Drawing Mode - Index finger is up
        # drawColor = (255,0,255)
        if fingers[1] and fingers[2] == False:
            drawColor = (255,0,255)
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # if drawColor == (0, 0, 0):
            #     cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
            #     cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            
            # else:
            #     cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            #     cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)


            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

        if fingers[1] and fingers[2] and fingers[3] == False:
            xp, yp = 0,0
            drawColor = (255,255,255)
            print('off mode')

        # elif fingers[1] == False and fingers[2]:
        #     drawColor = (0,0,0)
        #     print('eraser')
        #     if xp == 0 and yp == 0:
        #         xp, yp = x1, y1

        #     cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

        #     xp, yp = x1, y1

       
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    # print(img.shape)
    # print(imgCanvas.shape)
    # print(imgInv.shape)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)



    
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()