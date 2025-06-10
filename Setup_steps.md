## Setup and Inference Steps

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights**
   ```bash
   cd weights
   python get_weights.py  # Downloads resnet18.pkl
   cd ..
   ```

4. **Convert model to ONNX format**
   ```bash
   python model_to_onnx.py
   python pkl_to_onnx.py
   ```

5. **Convert ONNX model to OpenVINO IR (XML) format**
   ```bash
   mo --input_model <model.onnx>
   ```

6. **Run inference scripts**

   - **Original inference script**
     ```bash
     python inference.py
     ```

   - **OpenVINO API scripts**
     ```bash
     cd inference_scripts
     python openvino_xml.py
     python inference_openvino_api.py <onnx_model_path>
     ```