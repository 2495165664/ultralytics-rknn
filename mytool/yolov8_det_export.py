from ultralytics import YOLO
import cv2
# Load a model
model = YOLO('yolov8n.pt')
# model = YOLO('path/to/best.pt')  # load a custom model

model.export(format='onnx', opset=12, simplify=True, imgsz=320)

# Predict with the model
# results = model(r'D:\Users\wl\Desktop\Codes\ultralytics-yolov8\ultralytics\assets')  # predict on an image
# model.predict(r'D:\Users\wl\Desktop\Codes\ultralytics-yolov8\ultralytics\assets', save=True, imgsz=320, conf=0.3)
# print(results)
# for result in results:
#     # masks = result.masks
#     # print(result.boxes)
#     print(result.masks.shape)
#     print((result.masks))
#     # cv2.imshow("win", result.masks.cpu().numpy())
#     # cv2.waitKey(0)