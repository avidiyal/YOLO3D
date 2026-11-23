# Benchmarking Guide

This document explains how to use the provided scripts to benchmark inference performance for different execution modes: native PyTorch, ONNX models with OpenVINO Runtime, and OpenVINO IR models with OpenVINO Runtime.

All benchmark scripts are located in the `inference_scripts/` directory. They report average inference time in milliseconds (ms) and Frames Per Second (FPS). Higher FPS and lower ms indicate better performance.

**Prerequisites:**
- Ensure you have followed the setup steps in `README_openvino.md`, including environment setup, dependency installation, and model downloading/conversion.
- Use `weights/convert_models.py` to convert the source checkpoints directly to OpenVINO IR in `weights/openvino/`. By default it does not produce ONNX files. The optional ONNX benchmarks need ONNX models: run `python weights/convert_models.py --onnx` to also export them to `weights/onnx/`.

## Full 3D Detection Pipeline Benchmark (`benchmark_full_pipeline.py`)

This script benchmarks in-memory frame processing: 2D detection, NMS, crop preprocessing, 3D orientation/dimension regression, output decoding, geometric reconstruction, and drawing. Images and calibration matrices are preloaded; disk reads, model compilation, warm-up, and optional image saving are excluded from measured frame times.

- **Purpose:** To measure the end-to-end performance of the 3D object detection pipeline with different model backends (native PyTorch, ONNX with OpenVINO Runtime, OpenVINO IR with OpenVINO Runtime).
- **Models Expected:**
    - A 2D detector model (PyTorch `.pt`, `.onnx`, or OpenVINO IR `.xml`).
    - A 3D regressor model (PyTorch `.pkl`, `.onnx`, or OpenVINO IR `.xml`).
- **Key Arguments:**
    - `--pipeline_type`: Specifies the backend for both models (e.g., `pytorch_native`, `onnx_openvino`).
    - `--yolo_weights`, `--yolo_cfg` (if applicable).
    - `--regressor_weights`, `--regressor_name` (e.g., 'resnet18').
    - `--dataset_path`: Path to input images (e.g., `eval/image_2/`).
    - `--calib_file`: A calibration file containing `P2:` or `P_rect_02:`, or a directory with a matching `<image-stem>.txt` file for each image. Use `eval/calib/` with the sample images for per-image calibration.
    - `--img_size_yolo`, `--img_size_regressor`.
    - `--regressor_batch_size`: Number of object crops per regressor inference (default 16), or the maximum with `--dynamic-shapes`. The best value depends on your hardware.
    - `--dynamic-shapes`: Use a dynamic regressor batch size (1 to `--regressor_batch_size`) for OpenVINO IR/ONNX models instead of padding partial batches to a fixed size. CPU and GPU support dynamic shapes; NPU support is [limited](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html#dynamic-shapes).
    - `--device`: Target device (CPU, GPU, NPU).
    - `--data`: Dataset YAML containing class names for OpenVINO models (default: `data/coco128.yaml`).
    - `--output_path`: Optionally save annotated measured frames outside the timed region.
    - `--warmup_runs`, `--benchmark_runs`: Number of warm-up and measured frames. The script cycles through the images in `--dataset_path`.
  - **Output:** Mean, median, and P95 frame latency, overall FPS, the average time per pipeline stage, and the average number of detections and 3D boxes per frame.

**Example Command (OpenVINO IR Pipeline):**
```bash
python inference_scripts/benchmark_full_pipeline.py --pipeline_type ir_openvino --yolo_weights weights/openvino/yolov5s.xml --regressor_weights weights/openvino/resnet18.xml --regressor_name resnet18 --dataset_path eval/image_2/ --calib_file eval/calib/ --device CPU --warmup_runs 3 --benchmark_runs 10
```

**Example Command (Native PyTorch Pipeline):**
```bash
python inference_scripts/benchmark_full_pipeline.py --pipeline_type pytorch_native --yolo_weights weights/yolov5s.pt --yolo_cfg models/yolov5s.yaml --regressor_weights weights/resnet18.pkl --regressor_name resnet18 --dataset_path eval/image_2/ --calib_file eval/calib/ --device cpu --warmup_runs 3 --benchmark_runs 10
```

**Dynamic Batch Example:**
To benchmark dynamic batches, append `--dynamic-shapes` to an IR or ONNX command. For example:

```bash
python inference_scripts/benchmark_full_pipeline.py --pipeline_type ir_openvino --yolo_weights weights/openvino/yolov5s.xml --regressor_weights weights/openvino/resnet18.xml --dataset_path eval/image_2/ --calib_file eval/calib/ --device CPU --regressor_batch_size 4 --dynamic-shapes --warmup_runs 3 --benchmark_runs 10
```

Compare backends using the same shape mode and maximum batch size. Static mode pads partial batches, so static and dynamic results are not directly comparable.

**Example Command (OpenVINO ONNX Pipeline):**
This optional path requires YOLO and ResNet ONNX models. `weights/convert_models.py` only creates them when you add the `--onnx` flag: `python weights/convert_models.py --onnx`.

```bash
python inference_scripts/benchmark_full_pipeline.py --pipeline_type onnx_openvino --yolo_weights weights/onnx/yolov5s.onnx --yolo_cfg models/yolov5s.yaml --regressor_weights weights/onnx/resnet18.onnx --regressor_name resnet18 --dataset_path eval/image_2/ --calib_file eval/calib/ --device CPU --warmup_runs 3 --benchmark_runs 10
```

### Benchmarking INT8 Quantized Models

Once the OpenVINO IR models have been quantized to INT8 with `weights/quantize_yolo.py` and `weights/quantize_resnet.py` (see [Quantize models to INT8](README_openvino.md#5-quantize-models-to-int8-optional) in `README_openvino.md`, producing `weights/openvino/yolov5s_int8.xml` and `weights/openvino/resnet18_int8.xml`), benchmark them the same way as the regular OpenVINO IR pipeline above, just pointing `--yolo_weights`/`--regressor_weights` at the quantized files. The `--pipeline_type` stays `ir_openvino`.

**Example Command (INT8 OpenVINO IR Pipeline, CPU):**
```bash
python inference_scripts/benchmark_full_pipeline.py --pipeline_type ir_openvino --yolo_weights weights/openvino/yolov5s_int8.xml --regressor_weights weights/openvino/resnet18_int8.xml --regressor_name resnet18 --dataset_path eval/image_2/ --calib_file eval/calib/ --device CPU --warmup_runs 3 --benchmark_runs 10
```
Swap `--device CPU` for `GPU` or `NPU` to benchmark on other targets, exactly like the FP32 IR pipeline. Results are comparable directly against the FP32 IR run since both use the same `ir_openvino` pipeline type and inputs — only the model weights differ.

## Individual Model Benchmark Scripts

### 1. OpenVINO IR with OpenVINO Runtime (`benchmark_openvino_ir.py`)
   - **Purpose:** Measures inference speed of an OpenVINO Intermediate Representation (IR) model (`.xml` and `.bin` files) using the OpenVINO Runtime. This is often the most optimized path for OpenVINO.
  - **Model Expected:** OpenVINO IR model (`.xml` file with its matching `.bin` file). Convert the source checkpoint first using `weights/convert_models.py`, which saves IR models to `weights/openvino/`.
   - The optional `--dataset_path` argument can be used to specify a directory of images (e.g., `eval/image_2/`) or a single image file for benchmarking on real data. If not provided, the script uses randomly generated dummy data.
   - **Example Command:**
     ```bash
     python inference_scripts/benchmark_openvino_ir.py --model_path weights/openvino/yolov5s.xml --img_size 640 --batch_size 1 --device CPU --warmup_runs 10 --benchmark_runs 50 --dataset_path eval/image_2/
     ```

### 2. Native PyTorch (`benchmark_pytorch_native.py`)
   - **Purpose:** Measures the inference speed of a PyTorch model running natively on the specified device (CPU or CUDA) without any OpenVINO optimizations.
   - **Model Expected:** PyTorch model (`.pt` file) and its configuration (`.yaml` file).
   - The optional `--dataset_path` argument can be used to specify a directory of images (e.g., `eval/image_2/`) or a single image file for benchmarking on real data. If not provided, the script uses randomly generated dummy data.
   - **Example Command:**
     ```bash
     python inference_scripts/benchmark_pytorch_native.py --weights weights/yolov5s.pt --cfg models/yolov5s.yaml --img_size 640 --batch_size 1 --device cpu --warmup_runs 10 --benchmark_runs 50 --dataset_path eval/image_2/
     ```
     (Replace `cpu` with `cuda` if you have a compatible GPU and PyTorch GPU version installed.)

### 3. ONNX with OpenVINO Runtime (`benchmark_onnx.py`)
   - **Purpose:** Measures inference speed of an ONNX model using the OpenVINO Runtime.
  - **Model Expected:** An ONNX model (`.onnx` file). `weights/convert_models.py` produces IR only by default; add `--onnx` to also export ONNX models to `weights/onnx/`.
   - The optional `--dataset_path` argument can be used to specify a directory of images (e.g., `eval/image_2/`) or a single image file for benchmarking on real data. If not provided, the script uses randomly generated dummy data.
   - **Example Command:**
     ```bash
     python inference_scripts/benchmark_onnx.py --model_path weights/onnx/yolov5s.onnx --img_size 640 --batch_size 1 --device CPU --warmup_runs 10 --benchmark_runs 50 --dataset_path eval/image_2/
     ```

## Notes:
- **Device:** Performance will vary significantly based on the `--device` used (CPU, GPU, NPU) and the hardware capabilities.
- **First Run:** The first time you run an OpenVINO benchmark for a new model/device combination, there might be a one-time model compilation/optimization cost. Subsequent runs will be faster. The warm-up runs in the scripts help mitigate this in reported averages.
- **Benchmarking with Real Data vs. Dummy Data:** Using the `--dataset_path` option to benchmark with real images can provide more realistic performance figures. Dummy data is useful for quick checks and isolating pure computation speed.


