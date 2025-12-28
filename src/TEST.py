import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\code_chay\input\vehicle_count_input2.mp4")

TARGET_W = 1600
TARGET_H = 1000

ROI = np.array([
    (821, 435),
    (1250, 405),
    (1449, 479),
    (822, 555)
], np.int32)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (TARGET_W, TARGET_H))

    overlay = frame.copy()
    cv2.polylines(frame, [ROI], True, (0,0,255), 2)
    cv2.fillPoly(overlay, [ROI], (0,0,255))
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    cv2.imshow("ROI", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
