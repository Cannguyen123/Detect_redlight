from ultralytics import YOLO
import cv2
import numpy as np

import util
from Sort import *
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO(r'C:\code_chay\model\vehicle.pt')
license_plate_detector = YOLO(r'C:\code_chay\model\number_plate.pt')

# load video
cap = cv2.VideoCapture(r'C:\code_chay\input\demo.mp4')

vehicle_name_whitelist = [
    'Bus',
    'Cement mixer truck',
    'Garbage truck',
    'Pickup',
    'Pickup_modify',
    'SUV',
    'Sedan',
    'Tank truck',
    'Taxi',
    'Tow Truck',
    'Trailer truck',
    'Truck',
    'Van'
]

vehicles = [
    k for k, v in coco_model.names.items()
    if v in vehicle_name_whitelist
]

frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    results[frame_nmr] = {}

    # ================= VEHICLE DETECTION =================
    detections = coco_model(frame)[0]
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # ================= TRACKING =================
    # ================= TRACKING =================
    if len(detections_) > 0:
        track_ids = mot_tracker.update(np.asarray(detections_))
    else:
        track_ids = np.empty((0, 5))

    for track in track_ids:
        x1, y1, x2, y2, car_id = track.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {car_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ================= LICENSE PLATE DETECTION =================
    license_plates = license_plate_detector(frame)[0]

    for lp in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = lp

        # gán biển số cho xe
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)

        if car_id == -1:
            continue

        # crop biển số
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if license_plate_crop.size == 0:
            continue

        # OCR
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

        if license_plate_text is not None:
            # vẽ bbox biển số
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 0, 255), 2)
            cv2.putText(frame, license_plate_text,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)

            results[frame_nmr][car_id] = {
                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                'license_plate': {
                    'bbox': [x1, y1, x2, y2],
                    'text': license_plate_text,
                    'bbox_score': score,
                    'text_score': license_plate_text_score
                }
            }

    # ================= SHOW VIDEO =================
    cv2.imshow("Vehicle & License Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

# ================= SAVE RESULT =================
write_csv(results, './test.csv')

cap.release()
cv2.destroyAllWindows()
