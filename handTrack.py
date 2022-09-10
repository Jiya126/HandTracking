import time
import cv2
import mediapipe as mp

Mphands = mp.solutions.hands
hands = Mphands.Hands()

draw = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)

cTime = 0 
pTime = 0

while True:
    var, img = cam.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)   

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            draw.draw_landmarks(img, handLms, Mphands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx , cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (15,32), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)        

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()  