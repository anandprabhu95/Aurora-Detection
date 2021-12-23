import cv2
import numpy as np

# url = 'http://192.168.1.175:8080/video'
url = 'aurora.mp4'
cap = cv2.VideoCapture(url)
green_sens = 30


while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([60-green_sens, 100, 50])
    upper_green = np.array([60+green_sens, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow('frame', frame)
    # cv2.imshow('Preview', result)

    # Quit Program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    output = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.drawContours(output, contours, -1, 255, 3)
    countour_max = max(contours, key=cv2.contourArea)
    if cv2.contourArea(countour_max) > 3000:
        x, y, w, h = cv2.boundingRect(countour_max)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Aurora", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0))

    cv2.imshow("Preview", frame)
    cv2.resizeWindow("Preview", 1280, 780)

