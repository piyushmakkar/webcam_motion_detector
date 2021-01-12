import cv2,time

first_frame = None

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0) #iamge name, tuple h w,std deviation

    if first_frame is None:
        first_frame  = gray
        continue
    
    delta_frame = cv2.absdiff(first_frame,gray) 
    thresh_frame = cv2.threshold(delta_frame, 30,255,cv2.THRESH_BINARY)[1] #frame name, diff, colour , method # forthresh hold binary u need to access second item of the tuple
    thresh_frame = cv2.dilate(thresh_frame, None , iterations = 2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contours in cnts:
        if cv2.contourArea(contours) < 1000:
            continue

        (x,y,w,h) = cv2.boundingRect(contours)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow("gray",gray)
    cv2.imshow("delta",delta_frame)
    cv2.imshow("threshlod",thresh_frame)
    cv2.imshow("frame",frame)
   
    key=cv2.waitKey(1)
    
    if key == ord('r'):
        break

video.release()

cv2.destroyAllWindows()