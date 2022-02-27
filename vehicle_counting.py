### PROJECT = Using image processing methods for vehicle counting       date = 2019

import cv2
import numpy as np

video = cv2.VideoCapture('test.mp4')
delete_background = cv2.createBackgroundSubtractorMOG2()
morfoloji_kernel = np.ones((7,7),np.uint8)

class coordinate():
    def __init__(self,x,y):
        self.x = x
        self.y = y

class sensor():
    def __init__(self,coordinate1,coordinate2,width,length):
        self.coordinate1 = coordinate1
        self.coordinate2 = coordinate2
        self.width = width
        self.length = length
        self.mask = np.zeros((length,width,1),np.uint8)
        self.mask_region = abs(self.coordinate1.x - self.coordinate2.x)*abs(self.coordinate1.y - self.coordinate2.y)
        cv2.rectangle(self.mask ,(self.coordinate1.x,self.coordinate1.y),(self.coordinate2.x ,self.coordinate2.y),(255),thickness=cv2.FILLED)
        self.state = False
        self.left_detect_vehicle = 0
        self.right_detect_vehicle = 0

sensor1 = sensor(coordinate(290,60),coordinate(360,65),600,220)
sensor2 = sensor(coordinate(200,60),coordinate(270,65),600,220)

while (1):
    ret,kare =video.read()
    region_of_interest = kare[100:300,100:700]  
    non_background =delete_background.apply(region_of_interest)   
    morfoloji =cv2.morphologyEx(non_background,cv2.MORPH_OPEN,morfoloji_kernel) 
    contours,hierarchy= cv2.findContours(morfoloji,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    roi_yedek = region_of_interest.copy()
    roi_yedek2 = np.zeros((region_of_interest.shape[0],region_of_interest.shape[1],1),np.uint8)  
    roi_yedek3 = np.zeros((region_of_interest.shape[0],region_of_interest.shape[1],1),np.uint8)

    mask_inv1 = cv2.resize(sensor1.mask, (600,200), interpolation = cv2.INTER_AREA)   
    mask_inv2 = cv2.resize(sensor2.mask, (600,200), interpolation = cv2.INTER_AREA)
    roi_yedek2 = cv2.resize(roi_yedek2, (600,200), interpolation = cv2.INTER_AREA)     
    roi_yedek3 = cv2.resize(roi_yedek3, (600,200), interpolation = cv2.INTER_AREA)

    for i in contours:
        x,y,w,h = cv2.boundingRect(i)
        if w>30 and h>30:
            cv2.rectangle(roi_yedek,(x,y),(x+w,y+h),(255),thickness=2)
            cv2.rectangle(roi_yedek2,(x,y),(x+w ,y+h),(255),cv2.FILLED)
            cv2.rectangle(roi_yedek3,(x,y),(x+w ,y+h),(255),cv2.FILLED)

    sensor1_result = cv2.bitwise_and(roi_yedek2,roi_yedek2,mask=mask_inv1)
    sensor1_white_pixel_sum = np.sum(sensor1_result==255)
    sensor1_ratio = sensor1_white_pixel_sum / sensor1.mask_region

    sensor2_mask_sonuc = cv2.bitwise_and(roi_yedek3,roi_yedek3,mask=mask_inv2)
    sensor2_white_pixel_sum = np.sum(sensor2_mask_sonuc ==255)
    sensor2_ratio = sensor2_white_pixel_sum / sensor2.mask_region

    if (sensor1_ratio > 0.6 and sensor1.state==False):
        cv2.rectangle(roi_yedek,(sensor1.coordinate1.x , sensor1.coordinate1.y),(sensor1.coordinate2.x,sensor1.coordinate2.y),(0,255,0),thickness=cv2.FILLED)
        sensor1.state = True

    elif (sensor1_ratio < 0.4 and sensor1.state ==True):
        cv2.rectangle(roi_yedek,(sensor1.coordinate1.x , sensor1.coordinate1.y),(sensor1.coordinate2.x,sensor1.coordinate2.y),(0,0,255),thickness=cv2.FILLED)
        sensor1.state = False
        sensor1.right_detect_vehicle +=1
    else:
        cv2.rectangle(roi_yedek,(sensor1.coordinate1.x , sensor1.coordinate1.y),(sensor1.coordinate2.x,sensor1.coordinate2.y),(0,0,255),thickness=cv2.FILLED)


    if (sensor2_ratio > 0.6 and sensor2.state==False):
        cv2.rectangle(roi_yedek,(sensor2.coordinate1.x , sensor2.coordinate1.y),(sensor2.coordinate2.x,sensor2.coordinate2.y),(0,255,0),thickness=cv2.FILLED)
        sensor2.state = True

    elif (sensor2_ratio < 0.4 and sensor2.state ==True):
        cv2.rectangle(roi_yedek,(sensor2.coordinate1.x , sensor2.coordinate1.y),(sensor2.coordinate2.x,sensor2.coordinate2.y),(0,0,255),thickness=cv2.FILLED)
        sensor2.state = False
        sensor2.left_detect_vehicle +=1
    else:
        cv2.rectangle(roi_yedek,(sensor2.coordinate1.x , sensor2.coordinate1.y),(sensor2.coordinate2.x,sensor2.coordinate2.y),(0,0,255),thickness=cv2.FILLED)

    font =cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(roi_yedek,str(sensor1.right_detect_vehicle),(90,50),font,1,(255,255,255))
    cv2.putText(roi_yedek,str(sensor2.left_detect_vehicle),(30,50),font,1,(255,255,255))

    cv2.imshow('RESULT',roi_yedek)

    k=cv2.waitKey(30) & 0xff  
    if k==27:
        break

cv2.destroyAllWindows()
