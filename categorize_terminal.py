from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(source="0",stream=True)

for result in results:
    pass
