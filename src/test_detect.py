from ultralytics import YOLO

model = YOLO(r'/model/yolov8n.pt')
print(model.names)
