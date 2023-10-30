from ultralytics import YOLO
import cv2
# Load a model
model = YOLO('yolov8n-seg.pt')
# Predict with the model
results = model('./ultralytics/ultralytics/assets/zidane.jpg')  # predict on an image
res = results[0].plot(boxes=True) #boxes=False表示不展示预测框，True表示同时展示预测框
# Display the annotated frame
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)