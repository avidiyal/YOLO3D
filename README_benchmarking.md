# Benchmarking Guide

This document explains how to use the provided scripts to benchmark inference performance for different execution modes: native PyTorch, ONNX models with OpenVINO Runtime, and OpenVINO IR models with OpenVINO Runtime.

All benchmark scripts are located in the `inference_scripts/` directory. They report average inference time in milliseconds (ms) and Frames Per Second (FPS). Higher FPS and lower ms indicate better performance.

**Prerequisites:**
- Ensure you have followed the setup steps in `README_openvino.md`, including environment setup, dependency installation, and model downloading/conversion.
- Use `weights/convert_models.py` to convert the source checkpoints directly to OpenVINO IR in `weights/openvino/`. It does not produce ONNX files. The optional ONNX benchmarks require separately exported ONNX models. (Use convert_models.py with `--onnx` to produce them).

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
    - `--regressor_batch_size`: Fixed crop batch size by default, or maximum crop batch size with `--dynamic-shapes`. Compare batch sizes on your target hardware rather than assuming the default of 16 is optimal.
    - `--dynamic-shapes`: Opt into bounded dynamic crop batches (1 through `--regressor_batch_size`) for OpenVINO IR/ONNX models. Image dimensions and YOLO's single-frame batch remain fixed. Partial regressor batches run at their actual size, without padding, across all backends. A maximum of 1 naturally remains static. Without this flag, OpenVINO uses static shapes and all backends pad partial crop batches equally; only real crops are decoded and drawn. Shape support depends on the model and device; unsupported compilation fails rather than silently falling back to static shapes.
    - `--device`: Target device (CPU, GPU, NPU).
    - `--data`: Dataset YAML containing class names for OpenVINO models (default: `data/coco128.yaml`).
    - `--output_path`: Optionally save annotated measured frames outside the timed region.
    - `--warmup_runs`, `--benchmark_runs`: Counts of individual frames, cycling through the preloaded dataset. The regressor is also warmed independently, even if warm-up frames have no eligible detections. Dynamic mode performs `--warmup_runs` regressor warm-ups at every batch size from 1 through `--regressor_batch_size`; static mode warms only the fixed batch size. Zero disables both kinds of warm-up.
  - **Output:** Mean, median, and P95 frame wall-clock latency, FPS derived from total measured frame time, component timings, and separate counts of detections, crops, batches, skipped crops, and drawn 3D boxes. OpenVINO request latencies are reported separately and are not added again when computing FPS. A run with no eligible regressor crops warns that it measured detector-only work.

  OpenVINO uses reusable requests and application-owned input tensors. Native CUDA timings synchronize the device. The IR regressor receives raw uint8 crops without redundant host normalization; the ONNX and PyTorch regressors receive normalized crops.

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

Compare backends using the same shape mode and maximum batch size. Static and dynamic modes can execute different amounts of padded work, so their FPS values are not like-for-like model-throughput measurements. The standalone `benchmark_openvino_ir.py` retains its existing `--static-shapes` option.

**Example Command (OpenVINO ONNX Pipeline):**
This optional path requires separately exported YOLO and ResNet ONNX models; the default conversion step does not create them. Use convert_models.py with `--onnx` to create them.

```bash
python inference_scripts/benchmark_full_pipeline.py --pipeline_type onnx_openvino --yolo_weights weights/onnx/yolov5s.onnx --yolo_cfg models/yolov5s.yaml --regressor_weights weights/onnx/resnet18.onnx --regressor_name resnet18 --dataset_path eval/image_2/ --calib_file eval/calib/ --device CPU --warmup_runs 3 --benchmark_runs 10
```
(Adapt paths and `--pipeline_type` for other configurations like `pytorch_native`.)

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
  - **Model Expected:** A separately exported ONNX model (`.onnx` file). The default `weights/convert_models.py` workflow produces IR only, so provide an existing ONNX model for this optional benchmark.
   - The optional `--dataset_path` argument can be used to specify a directory of images (e.g., `eval/image_2/`) or a single image file for benchmarking on real data. If not provided, the script uses randomly generated dummy data.
   - **Example Command:**
     ```bash
     python inference_scripts/benchmark_onnx.py --model_path weights/onnx/yolov5s.onnx --img_size 640 --batch_size 1 --device CPU --warmup_runs 10 --benchmark_runs 50 --dataset_path eval/image_2/
     ```

## Notes:
- **Device:** Performance will vary significantly based on the `--device` used (CPU, GPU, NPU) and the hardware capabilities.
- **First Run:** The first time you run an OpenVINO benchmark for a new model/device combination, there might be a one-time model compilation/optimization cost. Subsequent runs will be faster. The warm-up runs in the scripts help mitigate this in reported averages.
- **Benchmarking with Real Data vs. Dummy Data:** Using the `--dataset_path` option to benchmark with real images can provide more realistic performance figures. Dummy data is useful for quick checks and isolating pure computation speed.


