from ultralytics import YOLO
from ultralytics import settings


model = YOLO('yolov8n.pt')
dataset_name = "../../../../data/V3/dataset.yaml"

# Display model information (optional)
model.info()

# Train the models
results = model.train(data=dataset_name, epochs=100, imgsz=640, plots=True, pretrained=True, augment=False, close_mosaic=0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,	mosaic=0.0, mixup=0.0, copy_paste=0.0)


model.val(data='../../../../data/V3/val_testdata.yaml')