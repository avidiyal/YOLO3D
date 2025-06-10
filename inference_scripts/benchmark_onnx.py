import torch
import numpy as np
from time import perf_counter
from models.yolo import Model  # Import the YOLO model architecture
import openvino as ov

# Load YOLOv5 model from a .pt file
weights_path = "yolov5s.pt"  # Path to your YOLOv5 weights file

# Define the model architecture
model = Model(cfg='models/yolov5s.yaml')  # Ensure you have the correct YAML configuration file

dummy_input = torch.randn(1, 3, 224, 640)

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

print("Creating OpenVINO Runtime Core")
core = ov.Core()

# Export the model to ONNX
model_onnx_path = "yolov5s_new.onnx"
try:
    torch.onnx.export(
        model,
        dummy_input,
        model_onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"ONNX model exported to {model_onnx_path}")
except Exception as e:
    print(f"Error exporting ONNX model: {e}")
    exit()

# Read the ONNX model with OpenVINO
try:
    model_new = core.read_model(model_onnx_path)
    print("ONNX model read successfully by OpenVINO")
except Exception as e:
    print(f"Error reading ONNX model with OpenVINO: {e}")
    exit()

# Compile the model
try:
    compiled_model = core.compile_model(model_new, "CPU")
    print("Model compiled successfully by OpenVINO")
except Exception as e:
    print(f"Error compiling model with OpenVINO: {e}")
    exit()

# Create random input tensor
random_input = torch.tensor(np.random.rand(1, 3, 640, 640).astype(np.float32))  # Adjust input size as needed
random_input = random_input.clone()

# Measure inference time
start = perf_counter()
try:
    results = compiled_model.infer_new_request({0: random_input.numpy()})  # Pass input as a dictionary
    print("Inference ran successfully")
except Exception as e:
    print(f"Error during inference: {e}")
    exit()
end = perf_counter()

avg_inf_time = 0
sum_inf_time = 0

for i in range(200):
    start = perf_counter()
    try:
        results = compiled_model.infer_new_request({0: random_input.numpy()})  # Pass input as a dictionary
    except Exception as e:
        print(f"Error during inference: {e}")
        exit()
    end = perf_counter()
    print(f"Time for 1 iteration is {end - start} s")
    sum_inf_time += (end - start)

avg_inf_time = sum_inf_time / 200
print(f"Average Inference Time: {avg_inf_time * 1000} ms")