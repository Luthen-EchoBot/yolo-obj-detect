import cv2
import numpy as np
# from sys import exit
from time import time,sleep
# from random import randint
# from PIL import Image

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(source="0",stream=True)

def draw_boxes(result):
    img = result.orig_img
    names = result.names
    boxes = result.boxes.xyxy.numpy().astype(np.int32)
    conf = result.boxes.conf.numpy()
    classes = result.boxes.cls.numpy()
    for score,cls,box in zip(conf,classes,boxes):
        color = (0,0,255) #(randint(0,255),randint(0,255),randint(0,255))
        class_label = names[cls]
        label=f"{class_label}: {score:0.2f}"
        lbl_margin = 3
        img = cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),thickness=1,color=color);
        label_size = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)
        lbl_w, lbl_h = label_size[0] # label w and h
        lbl_w += 2* lbl_margin # add margins on both sides
        lbl_h += 2*lbl_margin
        img = cv2.rectangle(img, (box[0], box[1]),(box[0]+lbl_w, box[1]-lbl_h),color=color,thickness=-1)
        cv2.putText(img, label, (box[0]+ lbl_margin,box[1]-lbl_margin),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0, color=(255, 255, 255 ),thickness=1)
    return img


# start = time()
for result in results:
    img = draw_boxes(result)
    cv2.imshow('img',img)
    if cv2.waitKey(25) & 0xff == ord('q'): # ms, break on 'q' pressed
        break
    # end = time()
    # if end-start > 10:
    #     break

cv2.destroyAllWindows()
