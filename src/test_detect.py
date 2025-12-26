from ultralytics import YOLO
import cv2

# load model biển số
license_plate_detector = YOLO(r'C:\code_chay\model\number_plate.pt')

# load video
cap = cv2.VideoCapture(r'C:\code_chay\input\demo.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detect biển số (GIẢM CONF)
    results = license_plate_detector(frame, conf=0.2)[0]

    for lp in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = lp

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 0, 255), 2)
        cv2.putText(frame, f'LP {score:.2f}',
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

    cv2.imshow("TEST LICENSE PLATE ONLY", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
