from ultralytics import YOLO
import cv2

VEHICLE_CLASSES = [
    'Bus', 'Truck', 'Taxi', 'Sedan', 'SUV',
    'Pickup', 'Van', 'Trailer truck'
]

cap = cv2.VideoCapture(r'/demo.mp4')

model_vehicle = YOLO('../vehicle.pt')
model_plate = YOLO('../best.pt')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = model_vehicle(frame)
    detections_vehicle = []

    for d in detections[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = d

        label = model_vehicle.names[int(class_id)]

        if label in VEHICLE_CLASSES:
            detections_vehicle.append(
                [x1, y1, x2, y2, conf, int(class_id)]
            )

        #tracking_vehicle:








#detect vehicles
#tracking vehicle
#detect numberplate
#number crop
#gan  bien va xe
cap.release()
cv2.destroyAllWindows()