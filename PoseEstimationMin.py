import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture('1.mp4') #CAPTURING VIDEO

mpPose = mp.solutions.pose #MP MEANS MEDIA PIPE FROM IMPORTED
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

Ptime = 0 #PREVIOUS TIME
Ctime =0 #CURRENT TIME


while True:
    success,im = cap.read() #VIDEO READING AS IMAGES
    img = cv2.resize(im, (1080, 1080)) 

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #CONVERTING IMAGES INTO COLOR RGB 
    results = pose.process(imgRGB)  #CALLING MEDIAPIPE TO PROCESS
    
    if results.pose_landmarks:
        for id,lm in enumerate(results.pose_landmarks.landmark):  #GETTING ID AND POSITION OF LANDMARKS
            h,w,c = img.shape                                      #GETTING HEIGHT WIDTH CHANNEL OF OUR IMG TO CONVERT
            cx,cy = int(lm.x*w),int(lm.y*h)                         #POSITIONS WE GOT FROM THE FOR LOOP INTO PIXELS
            print(id,cx,cy)
            # if id ==0:
            cv2.circle(img,(cx,cy),8,(0,255,0),cv2.FILLED)   #CIRCLE IN ID 0
                
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        

    # if results.multi_hand_landmarks:  #CHECKING FOR MULTIPPLE HANDS
    #     for handLms in results.multi_hand_landmarks:    # FOR EACH HAND

    #         for id,lm in enumerate(handLms.landmark):  #GETTING ID AND POSITION OF LANDMARKS
    #             h,w,c = img.shape                      #GETTING HEIGHT WIDTH CHANNEL OF OUR IMG TO CONVERT
    #             cx,cy = int(lm.x*w),int(lm.y*h)        #POSITIONS WE GOT FROM THE FOR LOOP INTO PIXELS
    #             print(id,cx,cy)
    #             if id ==0:
    #                 cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)   #CIRCLE IN ID 0
    #         mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)
    
    Ctime = time.time()
    fps = 1/(Ctime - Ptime)
    Ptime = Ctime
    
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("detection App (press q to exit)", img)
    key = cv2.waitKey(1)


    #key q pressed quit
    if(key == 81 or key == 113):
        break



