import torch
import numpy as np
from time import perf_counter
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.yolo import Model  # Import the YOLO model architecture
import openvino.torch
import openvino as ov
import sys
import os

# Load YOLOv5 model from a .pt file
weights_path = "../yolov5s.pt"  # Path to your YOLOv5 weights file

# Define the model architecture
model = Model(cfg='../../models/yolov5s.yaml')  # Ensure you have the correct YAML configuration file

dummy_input = torch.randn(1, 3, 640, 640)

state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)  # Load weights to CPU

# Check the type of the loaded object
print(type(state_dict))

# If the loaded object is a model instance, it might be incorrectly saved
if isinstance(state_dict, Model):
    print("The weights file contains a model instance, not a state dictionary.")
else:
    # Load the state dictionary into the model
    try:
        model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore missing keys
        print("YOLOv5 model weights loaded successfully.")
    except RuntimeError as e:
        print("Error loading YOLOv5 model weights:", e)
# Ensure the model is in evaluation mode
model.eval()

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "yolov5s_inference.onnx", opset_version=12)
