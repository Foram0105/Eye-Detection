import cv2

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

original_image = cv2.imread("Image.jpg")

# resize image
resizeimage = cv2.resize(original_image,(500,500))

# gray
grayimage = cv2.cvtColor(resizeimage,cv2.COLOR_BGR2GRAY)

eyes = eye_cascade.detectMultiScale(grayimage,scaleFactor=1.05,minNeighbors=5,minSize=(30,30),maxSize=(100,100))

if len(eyes)>=2:
    for(x,y,w,h) in eyes[:2]:
        cv2.rectangle(resizeimage, (x, y), (x + w, y + h), (0, 0, 0), 2)

cv2.imshow("Eye Detection",resizeimage)
cv2.waitKey(0)
cv2.destroyAllWindows()