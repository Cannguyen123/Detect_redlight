import cv2
from ultralytic import YOLO

#load video
cap=r'C:\code_chay\demo.mp4'
#load model:
model_vehicle=YOLO('vehicle.pt')
model_plate=YOLO('best.pt')
#read video
while True:
    ret,frame=cap.read()
    if not ret:
        break
    detections=model_vehicle(frame)
    for detection in detections.data.Tolist():
        


#detect vehicles
#tracking vehicle
#detect numberplate
#number crop
#gan  bien va xe
