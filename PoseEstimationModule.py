import cv2
import time
import mediapipe as mp


class poseDetector():
    def __init__(self, static_img= False, model_complexity =1, upper_body=False,
                 smooth =True, detection_confidence =0.5, tracking_confidence=0.5):
        self.static_img = static_img
        self.model_complexity = model_complexity
        self.upper_body = upper_body
        self.smooth = smooth
        self.detectionConfidence = detection_confidence
        self.trackingConfidence = tracking_confidence
        
        self.mpPose = mp.solutions.pose #MP MEANS MEDIA PIPE FROM IMPORTED
        self.pose = self.mpPose.Pose(self.static_img,self.model_complexity,self.upper_body,self.smooth,
                                     self.detectionConfidence,self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findPose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #CONVERTING IMAGES INTO COLOR RGB 
        self.results = self.pose.process(imgRGB)  #CALLING MEDIAPIPE TO PROCESS
        
        if self.results.pose_landmarks: 
            if draw:   
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)           
        return img    
    
    
    def findPosition(self,img,draw = True):
        lmList = []
        if self.results.pose_landmarks: 
            for id,lm in enumerate(self.results.pose_landmarks.landmark):  #GETTING ID AND POSITION OF LANDMARKS
                h,w,c = img.shape                      #GETTING HEIGHT WIDTH CHANNEL OF OUR IMG TO CONVERT
                cx,cy = int(lm.x*w),int(lm.y*h)        #POSITIONS WE GOT FROM THE FOR LOOP INTO PIXELS
                # print(id,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(0,255,0),cv2.FILLED)   #CIRCLE IN ID 0
        return lmList

def main():
    detector = poseDetector()
    cap = cv2.VideoCapture('1.mp4') #CAPTURING VIDEO
    
    Ptime = 0 #PREVIOUS TIME
    Ctime =0 #CURRENT TIME

    while True:
        success,im = cap.read() #VIDEO READING AS IMAGES
        img = cv2.resize(im,(1080,1024))
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) !=0:
            print(lmList[8])
        
        Ctime = time.time()
        fps = 1/(Ctime - Ptime)
        Ptime = Ctime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cv2.imshow("detection App (press q to exit)", img)
        key = cv2.waitKey(1)
        
        #key q pressed quit
        if(key == 81 or key == 113):
            break


if __name__ == "__main__":
    main()





