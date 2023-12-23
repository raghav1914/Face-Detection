import cv2

cap = cv2.VideoCapture(0)
status, photo = cap.read()

facemodel = cv2.CascadeClassifier("haarcascade_face.xml")

while True:
    status, pics = cap.read()
    myfacecoord = facemodel.detectMultiScale(pics)
    
    for (x, y, w, h) in myfacecoord:
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        cv2.rectangle(pics, (x1, y1), (x2, y2), [0, 255, 0], 2)
        
        # Display the cropped face region in a separate window
        face_roi = pics[y1:y2, x1:x2]
        cv2.imshow("Detected Face", face_roi)

    cv2.imshow("Webcam", pics)
    
    if cv2.waitKey(10) == 13:  # Wait for Enter key to exit
        break

cap.release()
cv2.destroyAllWindows()
