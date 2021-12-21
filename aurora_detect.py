import cv2
import numpy as np

url = 'http://192.168.0.5:8080/video'
cap = cv2.VideoCapture(url)
green_sens = 30
while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([60-green_sens, 100, 100])
    upper_green = np.array([60+green_sens, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

