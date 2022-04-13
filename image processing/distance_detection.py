import cv2
import numpy as np
import math




low=np.array([0,50,123])
high=np.array([255,255,255])

def findCnt(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low, high)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]

    return cnt


def updateHorizonatal(width,cnt):
    area = cv2.contourArea(cnt)
    M = cv2.moments(cnt)
    Cx = int(M['m10'] / M['m00'])
    Cy = int(M['m01'] / M['m00'])
    a = Cx
    b = Cy
    x, y, w, h = cv2.boundingRect(cnt)
    width=width/2
    width=int(width)
    horizontalDistance=a-width
    return horizontalDistance
  
def updateFront(cnt):
    area = cv2.contourArea(cnt)
    M = cv2.moments(cnt)
    Cx = int(M['m10'] / M['m00'])
    Cy = int(M['m01'] / M['m00'])
    a = Cx
    b = Cy
    cv2.circle(frame, (a, b), 3, (0, 0, 0), -1)
    x, y, w, h = cv2.boundingRect(cnt)
    distance=h/(2*math.tan(0.303))
    frontDistance=(distance*36.5)/h
    return frontDistance



# define a video capture object
vid = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')   #either(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
currentframe = 0


while(True):
    
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    height,width=frame.shape[0:2]
    cnt=findCnt(frame)
    horizontalDistance= updateHorizonatal(width,cnt)
    frontDistance= updateFront(cnt)
    print(horizontalDistance," ",frontDistance)
    S1= "frontDistance: "+str(frontDistance)
    S2= "horizontalDistance: "+str(horizontalDistance)
    cv2.putText(frame,S1, (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame,S2, (5, 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    out.write(frame) 
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# After the loop release the cap object
vid.release()



    

cv2.destroyAllWindows()
