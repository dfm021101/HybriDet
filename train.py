import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from pathlib import Path
from ultralytics import YOLO


# Path to the configuration file
config_path = "/home/oyaming/code/HybriDet/ultralytics/cfg/models/v8/HybriDet.yaml"

# Initialize YOLOv8n model
model = YOLO(config_path)

# Train the model
model.train(
    imgsz=640,  
    epochs=1000,
    batch=64,
    resume=False,
    save=True,
    workers=8,
    project="HybriDet_VOC",
    name="HybriDet",
    data="/home/oyaming/code/HybriDet/ultralytics/cfg/datasets/VOC.yaml",
    save_period=1,
    scale=False,
    single_cls=False,
    verbose=True,
    device='2',
)




