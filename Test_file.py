from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/26/yolo26_earlyexit.yaml")
print(model.model)