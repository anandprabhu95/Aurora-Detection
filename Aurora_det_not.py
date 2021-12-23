from urllib.request import urlopen
import cv2
import numpy as np
import winsound
from playsound import playsound
from pygame import mixer
import time


url = r'D:\Aurora_detection_notification\Aurora-Detection\Aurora_video\Warning.mp4'

cap = cv2.VideoCapture(url)
while True:
    # image_response = urlopen(url)
    ret, frame = cap.read()
    # cv2.imshow('temp', cv2.resize(image, (600, 400)))
    q = cv2.waitKey(1)
    if q == ord('q'):
        break
    

    # Convert the imageFrame in 
    # BGR(RGB color space) to 
    # HSV(hue-saturation-value)
    # color space

    # hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Set range for red color and 
    # define mask
    # red_lower = np.array([136, 87, 111], np.uint8)
    # red_upper = np.array([180, 255, 255], np.uint8)
    # red_mask = cv2.inRange(hsv, red_lower, red_upper)
  
    # Set range for green color and 
    # define mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
  
    # Set range for blue color and
    # define mask
    # blue_lower = np.array([94, 80, 2], np.uint8)
    # blue_upper = np.array([120, 255, 255], np.uint8)
    # blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

        # For red color
    # red_mask = cv2.dilate(red_mask, kernal)
    # res_red = cv2.bitwise_and(frame, frame, 
    #                           mask = red_mask)
      
    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(frame, frame,
                                mask = green_mask)
    
    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    output = cv2.bitwise_and(frame, frame, mask=green_mask)
    cv2.drawContours(output, contours, -1, 255, 3)
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    area_bb = w * h

    # draw the biggest contour (c) in green

    cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

# show the images
    cv2.imshow("Result", output)

    if area_bb / (720 * 1280) > 0.5:
        mixer.init() #Initialzing pyamge mixer
        mixer.music.load(r'C:\Users\shrey\Downloads\Steve-Vai-Tender-Surrender.mp3') #Loading Music File
        mixer.music.play() #Playing Music with Pygame
        time.sleep(3)
        mixer.music.stop()
        # playsound(r'C:\Users\shrey\Downloads\Steve-Vai-Tender-Surrender.mp3')

    # for pic, contour in enumerate(contours):
    #     print(type(contour))
    #     break
    #     area = cv2.contourArea(contour)
    #     if(area > 300):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         frame = cv2.rectangle(frame, (x, y), 
    #                                    (x + w, y + h),
    #                                    (0, 255, 0), 2)
              
    #         cv2.putText(frame, "Green Colour", (x, y),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 
    #                     1.0, (0, 255, 0))
    # For blue color
    # blue_mask = cv2.dilate(blue_mask, kernal)
    # res_blue = cv2.bitwise_and(frame, frame,
    #                            mask = blue_mask)

    # cv2.imshow('frame', frame)
    # Creating contour to track red color
    # contours, hierarchy = cv2.findContours(red_mask,
    #                                        cv2.RETR_TREE,
    #                                        cv2.CHAIN_APPROX_SIMPLE)

    # for pic, contour in enumerate(contours):
    #     area = cv2.contourArea(contour)
    #     if(area > 300):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         image = cv2.rectangle(image, (x, y), 
    #                                    (x + w, y + h), 
    #                                    (0, 0, 255), 2)
              
    #         cv2.putText(image, "Red Colour", (x, y),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0,
    #                     (0, 0, 255))    

        # Creating contour to track green color
    # contours, hierarchy = cv2.findContours(green_mask,
    #                                        cv2.RETR_TREE,
    #                                        cv2.CHAIN_APPROX_SIMPLE)
      
    # for pic, contour in enumerate(contours):
    #     area = cv2.contourArea(contour)
    #     if(area > 300):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         image = cv2.rectangle(image, (x, y), 
    #                                    (x + w, y + h),
    #                                    (0, 255, 0), 2)
              
    #         cv2.putText(image, "Green Colour", (x, y),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 
    #                     1.0, (0, 255, 0))
  
    # Creating contour to track blue color
    # contours, hierarchy = cv2.findContours(blue_mask,
    #                                        cv2.RETR_TREE,
    #                                        cv2.CHAIN_APPROX_SIMPLE)
    # for pic, contour in enumerate(contours):
    #     area = cv2.contourArea(contour)
    #     if(area > 300):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         image = cv2.rectangle(image, (x, y),
    #                                    (x + w, y + h),
    #                                    (255, 0, 0), 2)
              
    #         cv2.putText(image, "Blue Colour", (x, y),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     1.0, (255, 0, 0))


    # cv2.imshow("Aurora", image)
    # cv2.resizeWindow("Aurora", 1280, 780)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break



