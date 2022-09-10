import cv2
import time
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detConfidence=0.5, trConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detConfidence = detConfidence
        self.trConfidence = trConfidence

        self.Mphands = mp.solutions.hands
        self.hands = self.Mphands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detConfidence, self.trConfidence)
        self.draw = mp.solutions.drawing_utils

    def detectHands(self, img, drawY=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if drawY:
                    self.draw.draw_landmarks(img, handLms, self.Mphands.HAND_CONNECTIONS)

        return img

    def detectPos(self, img, handNo=0, drawY=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                    h,w,c = img.shape
                    cx , cy = int(lm.x*w), int(lm.y*h)
                    # print(id,cx,cy)
                    lmList.append([id,cx,cy])
                    if drawY:
                        cv2.circle(img, (cx,cy), 10, (255,255,0), cv2.FILLED)

        return lmList                

     



def main():
    cTime = 0 
    pTime = 0

    cam = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        var, img = cam.read()

        img = detector.detectHands(img, drawY=False)  
        lmList = detector.detectPos(img, drawY=False)
        if len(lmList) != 0:
            print(lmList)     

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (15,32), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)        

        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == "__main__":
    main()