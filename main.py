import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response

# ----------------------------
# Cargar modelo YOLO
# ----------------------------
model = YOLO("yolo11n.pt")

print("pre track")
results = model.track(source="0", stream=True, persist=True, classes=[0])
print("post track")

# ----------------------------
# Función opcional para dibujar cajas
# ----------------------------
def draw_boxes(result):
    img = result.orig_img
    names = result.names
    boxes = result.boxes.xyxy.numpy().astype(np.int32)
    conf = result.boxes.conf.numpy()
    classes = result.boxes.cls.numpy()
    for score, cls, box in zip(conf, classes, boxes):
        color = (0, 0, 255)
        class_label = names[cls]
        label = f"{class_label}: {score:0.2f}"
        lbl_margin = 3
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                            thickness=1, color=color)
        label_size = cv2.getTextSize(
            label, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, thickness=1
        )
        lbl_w, lbl_h = label_size[0]
        lbl_w += 2 * lbl_margin
        lbl_h += 2 * lbl_margin
        cv2.putText(img, label,
                    (box[0] + lbl_margin, box[1] + lbl_margin + lbl_h),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(255, 255, 255), thickness=1)
    return img

# ----------------------------
# Streaming Flask
# ----------------------------
app = Flask(__name__)

def generate_stream():
    for result in results:
        img = result.plot()  # O draw_boxes(result) si prefieres
        ret, jpeg = cv2.imencode('.jpg', img)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() +
               b'\r\n')

@app.route('/video')
def video():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Página principal
@app.route('/')
def index():
    return f'<h1>Streaming YOLO</h1><p>Abre <a href="/video">/video</a> para ver el streaming.</p>'

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Escucha en todas las interfaces, útil para túnel SSH o red local
    app.run(host="0.0.0.0", port=5000)
