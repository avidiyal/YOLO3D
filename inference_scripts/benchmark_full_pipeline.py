# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the YOLO3D pipeline across PyTorch and OpenVINO backends.

Before using --pipeline_type onnx_openvino, run from the repository root:
    python weights/convert_models.py --onnx

Use weights/onnx/yolov5s.onnx for --yolo_weights and weights/onnx/resnet18.onnx
for --regressor_weights. ONNX ResNet receives normalized float32 crops;
the exported OpenVINO ResNet IR receives raw uint8 crops instead.
"""

import argparse
import time
import numpy as np
import torch
from pathlib import Path
import sys
import collections # For defaultdict
import glob
import cv2
import yaml

# Add project root for custom module imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # Assuming this script is in inference_scripts/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from utils.datasets import LoadImages
    from inference_scripts.benchmarkable_detect3d_pipeline import (
        benchmarkable_detect3d, load_yolo_detector, load_regressor_model,
        load_projection, infer_openvino, synchronized_time,
    )
    from script.Dataset import ClassAverages, generate_bins # Assuming ClassAverages is in script.Dataset
    # from utils.torch_utils import select_device # Using torch.device directly
except ImportError as e:
    print(f"Error: Failed to import necessary modules: {e}. Ensure PYTHONPATH is correctly set.", file=sys.stderr)
    sys.exit(1)

def load_benchmark_frames(dataset_path, calibration_path):
    source = Path(dataset_path)
    files = sorted(Path(name) for name in glob.glob(str(source), recursive=True)) if '*' in str(source) else (
        sorted(source.iterdir()) if source.is_dir() else [source]
    )
    image_suffixes = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo'}
    video_suffixes = {'.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv'}
    if any(path.suffix.lower() in video_suffixes for path in files):
        images = [(Path(path), image) for path, _, image, _, _ in LoadImages(str(source), auto=False)]
    else:
        images = []
        for path in files:
            if path.suffix.lower() not in image_suffixes:
                continue
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not read image: {path}")
            images.append((path, image))
    if not images:
        raise ValueError(f"No images found at {dataset_path}")
    calibration_path = Path(calibration_path)
    projections = {}
    frames = []
    for path, image in images:
        calibration_file = calibration_path / f'{path.stem}.txt' if calibration_path.is_dir() else calibration_path
        if calibration_file not in projections:
            projections[calibration_file] = load_projection(calibration_file)
        frames.append((path, image, projections[calibration_file]))
    return frames


def summarize_timings(frame_timings):
    wall_times = np.asarray([frame['frame_wall_time'] for frame in frame_timings])
    return {
        'mean_ms': float(wall_times.mean() * 1000),
        'median_ms': float(np.median(wall_times) * 1000),
        'p95_ms': float(np.percentile(wall_times, 95) * 1000),
        'fps': float(len(wall_times) / wall_times.sum()) if wall_times.sum() > 0 else 0.0,
    }


def warmup_regressor(model, buffer, device, runs, request=None, dynamic_shapes=False):
    batch_sizes = range(1, buffer.shape[0] + 1) if dynamic_shapes else (buffer.shape[0],)
    for _ in range(runs):
        for batch_size in batch_sizes:
            inputs = buffer[:batch_size]
            if request is not None:
                infer_openvino(model, inputs, request)
            else:
                with torch.no_grad():
                    model(torch.from_numpy(inputs).to(device))
                synchronized_time(device)


def main():
    parser = argparse.ArgumentParser(
        description="Full 3D Detection Pipeline Benchmark Script",
        epilog="For onnx_openvino, first run: python weights/convert_models.py --onnx (from the repository root).",
    )
    parser.add_argument('--pipeline_type', type=str, required=True,
                        choices=['pytorch_native', 'onnx_openvino', 'ir_openvino'],
                        help="Type of pipeline to benchmark.")
    parser.add_argument('--yolo_weights', type=str, required=True, help="Path to 2D detector model file (e.g., .pt, .onnx, .xml).")
    parser.add_argument('--yolo_cfg', type=str, help="Path to YOLO model configuration YAML (required for PyTorch types).")
    parser.add_argument('--regressor_weights', type=str, required=True, help="Path to 3D regressor model file (e.g., .pkl, .onnx, .xml).")
    parser.add_argument('--regressor_name', type=str, default='resnet18', help="Name of the regressor model (e.g., 'resnet18').")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to input image dataset directory or single image file.")
    parser.add_argument('--calib_file', type=str, required=True, help="P2/P_rect_02 calibration file, or directory of image-stem .txt files.")
    parser.add_argument('--data', type=Path, default=ROOT / 'data/coco128.yaml', help="Dataset YAML containing class names for OpenVINO models.")
    parser.add_argument('--output_path', type=Path, help="Optionally save annotated measured frames outside the timed region.")

    parser.add_argument('--img_size_yolo', type=int, default=640, help="Target image size (height=width) for YOLO detector input.")
    parser.add_argument('--img_size_regressor', type=int, default=224, help="Target image size (height=width) for Regressor input patches.")

    # Note: Full pipeline processes one image at a time. YOLO batch size is effectively 1 here.
    # Regressor batch size is for processing multiple detected objects from a single image.
    parser.add_argument('--regressor_batch_size', type=int, default=16, help="Fixed crop batch size, or maximum crop batch size with --dynamic-shapes.")
    parser.add_argument('--dynamic-shapes', action='store_true', help="Compile OpenVINO with bounded dynamic crop batches (1 to regressor_batch_size) and skip padding; YOLO batch and image dimensions stay fixed.")

    parser.add_argument('--warmup_runs', type=int, default=5, help="Number of warm-up frames, plus regressor warm-ups per batch size (every size from 1 to the maximum in dynamic mode).")
    parser.add_argument('--benchmark_runs', type=int, default=20, help="Number of measured frames; cycles through preloaded images.")
    parser.add_argument('--device', type=str, default='cpu', help="Device: 'cpu' or 'cuda' for PyTorch native; 'CPU', 'GPU', etc. for OpenVINO backends/runtimes.")

    parser.add_argument('--conf_thres_yolo', type=float, default=0.25, help="Confidence threshold for YOLO NMS.")
    parser.add_argument('--iou_thres_yolo', type=float, default=0.45, help="IOU threshold for YOLO NMS.")
    parser.add_argument('--yolo_classes_to_detect', type=int, nargs='+', default=None, help="Filter by class: e.g., 0, or 0 2 3. Default: all classes.")
    parser.add_argument('--max_detections_yolo', type=int, default=1000, help="Maximum number of detections per image from YOLO.")

    args = parser.parse_args()
    if args.warmup_runs < 0 or args.benchmark_runs <= 0 or args.regressor_batch_size <= 0:
        parser.error("warmup_runs must be nonnegative; benchmark_runs and regressor_batch_size must be positive")
    if args.img_size_yolo <= 0 or args.img_size_regressor != 224:
        parser.error("img_size_yolo must be positive; the regressor requires img_size_regressor=224")

    # Validate conditional cfg path
    if args.pipeline_type == 'pytorch_native' and not args.yolo_cfg:
        print("Error: --yolo_cfg is required for PyTorch-based pipeline types.", file=sys.stderr)
        sys.exit(1)

    # --- Device Setup ---
    # For PyTorch native, device_str is 'cpu' or 'cuda'.
    # For OpenVINO ONNX/IR, device_str is OpenVINO device type ('CPU', 'GPU').
    pytorch_target_device_str = args.device if args.pipeline_type == 'pytorch_native' else 'cpu'
    pytorch_device = torch.device(pytorch_target_device_str)
    openvino_target_device_str = args.device.upper() # OpenVINO expects uppercase device names

    # --- Load Models ---
    print(f"--- Loading models for pipeline type: {args.pipeline_type} ---")
    yolo_model = load_yolo_detector(
        args.yolo_weights, args.yolo_cfg,
        device_str=args.device, # Pass the original device string
        model_type=args.pipeline_type,
        img_size=(args.img_size_yolo, args.img_size_yolo), # Pass as tuple
        batch_size=1, # YOLO processes one image at a time in this pipeline
        dynamic_shapes=args.dynamic_shapes,
    )
    if yolo_model is None: sys.exit("YOLO model loading failed.")

    regressor_model = load_regressor_model(
        args.regressor_weights, args.regressor_name,
        device_str=args.device, # Pass the original device string
        model_type=args.pipeline_type,
        img_size=(args.img_size_regressor, args.img_size_regressor), # Pass as tuple
        batch_size=args.regressor_batch_size, # Regressor can batch internal detections
        dynamic_shapes=args.dynamic_shapes,
    )
    if regressor_model is None: sys.exit("Regressor model loading failed.")

    # --- Load Utilities ---
    try:
        class_averages = ClassAverages() # Assumes 'class_dims.txt' is in working dir or findable
        angle_bins = generate_bins(2) # Example: 2 bins for orientation
    except Exception as e:
        print(f"Error loading ClassAverages or generating angle_bins: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Determine YOLO Stride and Class Names ---
    yolo_model_stride = 32
    yolo_class_names = [f'class_{i}' for i in range(80)] # Default
    if args.pipeline_type == 'pytorch_native':
        yolo_eval_model = yolo_model
        if hasattr(yolo_eval_model, 'stride'):
            yolo_model_stride = int(yolo_eval_model.stride.max().item() if isinstance(yolo_eval_model.stride, torch.Tensor) else yolo_eval_model.stride)
        if hasattr(yolo_eval_model, 'names'):
            yolo_class_names = yolo_eval_model.names
        if isinstance(yolo_class_names, dict):
            yolo_class_names = [yolo_class_names[index] for index in sorted(yolo_class_names)]
        print(f"Using YOLO Stride: {yolo_model_stride}, Class Names: {yolo_class_names[:3]}...")
    else: # For OpenVINO models, stride might need to be known or passed as arg. Using default.
        with args.data.open(encoding='utf-8') as stream:
            yolo_class_names = yaml.safe_load(stream)['names']
        if isinstance(yolo_class_names, dict):
            yolo_class_names = [yolo_class_names[index] for index in sorted(yolo_class_names)]
        print(f"Using YOLO stride {yolo_model_stride} and class names from {args.data}.")


    # --- Load Dataset ---
    print(f"Loading dataset from: {args.dataset_path}...")
    frames = load_benchmark_frames(args.dataset_path, args.calib_file)
    print(f"Loaded {len(frames)} frames and cached calibration. Disk I/O is excluded from timings.")
    openvino_pipeline = args.pipeline_type in ('onnx_openvino', 'ir_openvino')
    yolo_request = yolo_model.create_infer_request() if openvino_pipeline else None
    regressor_request = regressor_model.create_infer_request() if openvino_pipeline else None
    regressor_dtype = np.uint8 if args.pipeline_type == 'ir_openvino' else np.float32
    regressor_buffer = np.zeros((args.regressor_batch_size, 3, 224, 224), dtype=regressor_dtype)
    warmup_regressor(
        regressor_model, regressor_buffer, pytorch_device, args.warmup_runs,
        request=regressor_request, dynamic_shapes=args.dynamic_shapes,
    )

    # --- Warm-up Phase ---
    for i in range(args.warmup_runs):
        _, image_np_hwc, projection = frames[i % len(frames)]
        _, _ = benchmarkable_detect3d(
            yolo_model=yolo_model, regressor_model=regressor_model,
            yolo_model_type=args.pipeline_type, regressor_model_type=args.pipeline_type,
            input_image_np=image_np_hwc,
            img_size_yolo=(args.img_size_yolo, args.img_size_yolo), stride_yolo=yolo_model_stride,
            img_size_regressor=(args.img_size_regressor, args.img_size_regressor),
            device_pytorch=pytorch_device, device_openvino=openvino_target_device_str,
            calib_file_path=projection, class_averages=class_averages, angle_bins=angle_bins,
            conf_thres_yolo=args.conf_thres_yolo, iou_thres_yolo=args.iou_thres_yolo,
            yolo_classes_to_detect=args.yolo_classes_to_detect, max_detections_yolo=args.max_detections_yolo,
            regressor_batch_size=args.regressor_batch_size,
            override_class_names=yolo_class_names,
            yolo_request=yolo_request, regressor_request=regressor_request,
            regressor_buffer=regressor_buffer,
            dynamic_shapes=args.dynamic_shapes,
        )

    # --- Benchmark Phase ---
    overall_timings_list = []
    if args.output_path is not None:
        args.output_path.mkdir(parents=True, exist_ok=True)
    for i in range(args.benchmark_runs):
        image_path, image_np_hwc, projection = frames[i % len(frames)]
        frame_start = synchronized_time(pytorch_device)
        annotated_image, frame_timings = benchmarkable_detect3d(
            yolo_model=yolo_model, regressor_model=regressor_model,
            yolo_model_type=args.pipeline_type, regressor_model_type=args.pipeline_type,
            input_image_np=image_np_hwc,
            img_size_yolo=(args.img_size_yolo, args.img_size_yolo), stride_yolo=yolo_model_stride,
            img_size_regressor=(args.img_size_regressor, args.img_size_regressor),
            device_pytorch=pytorch_device, device_openvino=openvino_target_device_str,
            calib_file_path=projection, class_averages=class_averages, angle_bins=angle_bins,
            conf_thres_yolo=args.conf_thres_yolo, iou_thres_yolo=args.iou_thres_yolo,
            yolo_classes_to_detect=args.yolo_classes_to_detect, max_detections_yolo=args.max_detections_yolo,
            regressor_batch_size=args.regressor_batch_size,
            override_class_names=yolo_class_names,
            yolo_request=yolo_request, regressor_request=regressor_request,
            regressor_buffer=regressor_buffer,
            dynamic_shapes=args.dynamic_shapes,
        )
        frame_timings['frame_wall_time'] = synchronized_time(pytorch_device) - frame_start
        overall_timings_list.append(frame_timings)
        if args.output_path is not None:
            output_file = args.output_path / f'{i:05d}_{image_path.stem}.png'
            if not cv2.imwrite(str(output_file), annotated_image):
                raise RuntimeError(f"Could not save annotated frame: {output_file}")

    # --- Results Aggregation and Output ---
    if not overall_timings_list:
        print("No benchmark timings recorded.", file=sys.stderr)
        sys.exit(1)

    aggregated_timings = collections.defaultdict(list)
    for frame_time_dict in overall_timings_list:
        for key, value in frame_time_dict.items():
            aggregated_timings[key].append(value)

    duration_keys = [key for key in aggregated_timings if not key.startswith('num_') and key != 'frame_wall_time']
    avg_timings_ms = {
        key: np.mean([frame.get(key, 0.0) for frame in overall_timings_list]) * 1000
        for key in duration_keys
    }
    summary = summarize_timings(overall_timings_list)

    print("\n--- Full 3D Detection Pipeline Benchmark Results ---")
    print(f"Pipeline Type: {args.pipeline_type}")
    print(f"YOLO Model: {args.yolo_weights}")
    if args.yolo_cfg: print(f"YOLO Config: {args.yolo_cfg}")
    print(f"Regressor Model: {args.regressor_weights} (Type: {args.regressor_name})")
    print(f"Dataset: {args.dataset_path} ({len(frames)} preloaded frames used for cycling)")
    print("Measured: preprocessing, inference, output decoding, geometry, drawing and frame overhead; excludes disk I/O.")
    if args.pipeline_type == 'pytorch_native':
        print(f"PyTorch Native Device: {pytorch_device.type}")
    elif args.pipeline_type in ['onnx_openvino', 'ir_openvino']:
        print(f"Target OpenVINO Device: {openvino_target_device_str}")
    print(f"YOLO Input Size: {args.img_size_yolo}x{args.img_size_yolo}")
    print(f"Regressor Input Patch Size: {args.img_size_regressor}x{args.img_size_regressor}")
    print(f"Regressor Batch Size (for detected objects): {args.regressor_batch_size}")
    print(f"Shape mode: {'dynamic batch (unpadded)' if args.dynamic_shapes else 'static batch (padded)'}; spatial dimensions fixed")
    print(f"Warm-up Runs: {args.warmup_runs}, Benchmark Runs: {args.benchmark_runs}")
    print("----------------------------------------------------")
    print("Average Time per Component (ms):")
    for key, avg_ms in avg_timings_ms.items():
        print(f"  - {key.replace('_', ' ').capitalize()}: {avg_ms:.3f} ms")
    print("Average Counts per Frame:")
    for key, values in aggregated_timings.items():
        if key.startswith('num_'):
            label = 'Regressor crops' if key == 'num_regressor_runs' else key[4:].replace('_', ' ')
            print(f"  - {label}: {np.mean(values):.2f}")
    print("----------------------------------------------------")
    print(f"Mean Frame Wall Time: {summary['mean_ms']:.3f} ms")
    print(f"Median Frame Wall Time: {summary['median_ms']:.3f} ms")
    print(f"P95 Frame Wall Time: {summary['p95_ms']:.3f} ms")
    print(f"Overall Pipeline FPS: {summary['fps']:.2f}")
    if not sum(frame['num_regressor_runs'] for frame in overall_timings_list):
        print("Warning: no eligible crops reached the regressor; this run measured detector-only work.")
    print("----------------------------------------------------\n")

if __name__ == '__main__':
    main()
