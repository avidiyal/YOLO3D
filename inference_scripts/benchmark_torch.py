import torch
import numpy as np
from time import perf_counter
from models.yolo import Model  # Import the YOLO model architecture
import openvino.torch
import openvino as ov

# Load YOLOv5 model from a .pt file
weights_path = "yolov5s.pt"  # Path to your YOLOv5 weights file

# Define the model architecture
model = Model(cfg='models/yolov5s.yaml')  # Ensure you have the correct YAML configuration file

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
# torch.onnx.export(model, dummy_input, "yolov5s.onnx", opset_version=11)

# Compile the model with OpenVINO backend
model = torch.compile(model, backend="openvino", options={"device": "CPU"})

# Create random input tensor
random_input = torch.tensor(np.random.rand(1, 3, 640, 640).astype(np.float32))  # Adjust input size as needed
random_input = random_input.clone()

# Measure inference time
start = perf_counter()
with torch.no_grad():  # Disable gradient computation
    result = model(random_input)
end = perf_counter()

avg_inf_time = 0
sum_inf_time = 0

for i in range(200):
    start = perf_counter()
    with torch.no_grad():
        result = model(random_input)
    end = perf_counter()
    print(f"Time for 1 iteration is {end - start} s")
    sum_inf_time += (end - start)

avg_inf_time = sum_inf_time / 200
print(f"Average Inference Time: {avg_inf_time * 1000} ms")