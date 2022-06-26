import cv2
import firebase_admin
import time
from firebase_admin import credentials
from firebase_admin import firestore

import os

thres = 0.45 # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'


cred = credentials.Certificate(r"C:\Users\ayush\PycharmProjects\Object Detector\fb-info.json") # You would need to connect this to your own firebase server
firebase_admin.initialize_app(cred)

db = firestore.client()
doc_ref = db.collection(u'detector').document("live") # add this document to your own firebase server



net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

confidenceScore = 0 # if value is above 0.7, active punishment

efficencyMetric = 0

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    #print(classIds,bbox)

    if 77 in classIds or 73 in classIds or 61 in classIds:
        confidenceScore = 1
        if efficencyMetric == 1:
            field_updates = {"value": 1}
            doc_ref.update(field_updates)
        print("ALERT: Off Task! Cell Phone Detected with high confidence level")
        efficencyMetric = 0
    else:
        confidenceScore = 0
        if efficencyMetric == 0:
            field_updates = {"value": 0}
            doc_ref.update(field_updates)
        print("On Task: No cell phone or distractions detected")
        efficencyMetric = 1








    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Output",img)
    cv2.waitKey(1)