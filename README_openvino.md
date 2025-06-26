## Setup and Inference Steps

1. **Create a virtual environment**
   ```bash
   python -m venv ov-yolo3d-env
   source ov-yolo3d-env/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements-cpu.txt
   ```
   - For Intel GPU and NPU devices, make sure to [install appropriate device drivers](https://docs.openvino.ai/2025/get-started/install-openvino/configurations.html)

3. **Download model weights**
   This step downloads pre-trained model weights using the `get_weights.py` script, which should be run from the `weights/` directory. This will download `resnet18.pkl` and `yolov5s.pt` into the `weights/` directory.

   - **Default behavior (downloads ResNet18 PKL and YOLOv5s PT):**
     ```bash
     cd weights
     python get_weights.py
     ```

4. **Convert Source Models to ONNX and OpenVINO IR**

   This step uses the unified script `weights/convert_models.py` to convert  source models (e.g., PyTorch `.pt` files, `.pkl` files) into ONNX format and then into OpenVINO Intermediate Representation (IR) format (`.xml` and `.bin` files). Converted models are saved to `weights/onnx/` for ONNX files and `weights/openvino/` for OpenVINO IR files by default.

   - **Default Conversion (YOLOv5s and ResNet18):**
     To convert the default `yolov5s.pt` (using `models/yolov5s.yaml`) and `resnet18.pkl` (both expected to be in the `weights/` directory after Step 3), run:
     ```bash
     python convert_models.py
     ```
     This will produce:
     - `weights/onnx/yolov5s.onnx`
     - `weights/openvino/yolov5s.xml` and `weights/openvino/yolov5s.bin`
     - `weights/onnx/resnet18.onnx`
     - `weights/openvino/resnet18.xml` and `weights/openvino/resnet18.bin`

5. **Run inference scripts**

   - **Original inference script**
     (This script might need updates if it doesn't use the new model paths, or it might be deprecated if focus is on OpenVINO scripts.)
     ```bash
     python inference.py
     ```

   - **OpenVINO Accelerated Scripts**
     ```bash
     cd inference_scripts
     ```
     - To run an ONNX model directly using OpenVINO APIs:
       Example using `yolov5s.onnx`:
       ```bash
       # Assumes yolov5s.onnx was created in the previous step (4).
       python inference_openvino_api.py --save_result
       ```
     - To run a model with OpenVINO IR format (XML/BIN):
       Example using `yolov5s.xml` and `resnet18.xml`:
       ```bash
       # Assumes yolov5s.xml and resnet18.xml were created in step 4.
       python inference_openvino_xml.py  --save_result
       ```

     - To leverage `torch.compile` for an easier path to OpenVINO acceleration:
       Example using `yolov5s.pt`:
       ```bash
       # Assumes weights/yolov5s.pt is available
       python inference_openvino_compile.py --save_result
       ```

## Benchmarking Performance

For detailed instructions on how to run benchmarks, please refer to the [Benchmarking Guide](README_benchmarking.md).