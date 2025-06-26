# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np # For an unused variable, but often used with datasets
import time
import argparse
from pathlib import Path
import sys
import cv2 # Explicitly import cv2, as LoadImages might use it
import random # For selecting images from the preloaded list

# Add project root to sys.path to allow imports from models and utils packages
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from models.yolo import Model  # For loading YOLOv5 model structure
    from utils.datasets import LoadImages # For loading real datasets
except ImportError as e:
    print(f"Error: Could not import 'Model' from 'models.yolo' or 'LoadImages' from 'utils.datasets'. {e}", file=sys.stderr)
    print("Ensure the yolov5 project root is correctly added to sys.path and required modules exist.", file=sys.stderr)
    # Provide dummy classes if imports fail
    class Model:
        def __init__(self, cfg, ch=None, nc=None):
            raise ImportError("Real Model class not available.")
        def to(self, device): raise ImportError("Real Model class not available.")
        def load_state_dict(self, sd, strict=False): raise ImportError("Real Model class not available.")
        def eval(self): raise ImportError("Real Model class not available.")
        def __call__(self, x): raise ImportError("Real Model class not available.")
    class LoadImages:
        def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
            raise ImportError("Real LoadImages class not available.")
        def __iter__(self): return self
        def __next__(self): raise StopIteration
        def __len__(self): return 0


def main():
    parser = argparse.ArgumentParser(description="Native PyTorch Model Benchmark Script with Real Dataset Support")
    parser.add_argument('--weights', type=str, required=True, help='Path to PyTorch model weights file (e.g., weights/yolov5s.pt)')
    parser.add_argument('--cfg', type=str, required=True, help='Path to model configuration YAML file (e.g., models/yolov5s.yaml)')
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to dataset directory or image file. If None, uses dummy data.')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for model input (square image assumed)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--warmup_runs', type=int, default=10, help='Number of warm-up inference runs')
    parser.add_argument('--benchmark_runs', type=int, default=50, help='Number of benchmark inference runs')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to run on (cpu or cuda)')

    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"Error: Weights file not found at {args.weights}", file=sys.stderr)
        return
    if not Path(args.cfg).exists():
        print(f"Error: Model config file not found at {args.cfg}", file=sys.stderr)
        return
    if args.dataset_path and not Path(args.dataset_path).exists() and not Path(args.dataset_path).is_file():
         if not (Path(args.dataset_path).parent.exists() and '*' in Path(args.dataset_path).name): # check for glob patterns
            print(f"Error: Dataset path {args.dataset_path} does not exist or is not a file/glob pattern.", file=sys.stderr)
            return


    # Model Loading
    model_stride = 32 # Default stride
    model = None # Initialize model variable
    YOLOModel = Model # Alias for clarity in loading logic
    try:
        device = torch.device(args.device)
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA selected but not available. Defaulting to CPU.", file=sys.stderr)
            device = torch.device('cpu') # Update device if fallback occurs

        print(f"Loading PyTorch model from weights: '{args.weights}' onto device: {device}...")
        checkpoint = torch.load(args.weights, map_location=device, weights_only=False)

        final_model = None
        if hasattr(checkpoint, 'yaml') and hasattr(checkpoint, 'model') and hasattr(checkpoint, 'eval'): # Ultralytics full model
            print("Checkpoint appears to be a full model object (Ultralytics format).")
            final_model = checkpoint.to(device).eval()
            # For benchmarking, we generally want to respect args.cfg if provided,
            # to ensure the architecture matches what user expects to benchmark.
            # If cfg is different, re-load state_dict into model defined by args.cfg.
            # This step is optional if we trust the .pt file's baked-in cfg or if no args.cfg is used for this type.
            # However, this script requires args.cfg, so we align.
            if args.cfg:
                 print(f"Aligning loaded model with specified CFG: {args.cfg}")
                 temp_model_from_cfg = YOLOModel(cfg=args.cfg, ch=3, nc=None).to(device)
                 # model.state_dict() might be from a fused model, ensure it's compatible
                 temp_model_from_cfg.load_state_dict(final_model.state_dict())
                 final_model = temp_model_from_cfg.eval()


        elif isinstance(checkpoint, dict) and 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'):
            print("Checkpoint is a dictionary with a 'model' attribute (nn.Module).")
            loaded_sub_model = checkpoint['model']
            if hasattr(loaded_sub_model, 'yaml_file') or hasattr(loaded_sub_model, 'yaml'): # If it's a YOLOModel instance with its own config
                 final_model = loaded_sub_model.to(device).float().eval()
                 # Similar alignment with args.cfg if necessary
                 if args.cfg:
                    print(f"Aligning loaded sub-model with specified CFG: {args.cfg}")
                    temp_model_from_cfg = YOLOModel(cfg=args.cfg, ch=3, nc=None).to(device)
                    temp_model_from_cfg.load_state_dict(final_model.state_dict())
                    final_model = temp_model_from_cfg.eval()
            else: # Assume it's a more generic nn.Module, load its state_dict into our cfg model
                 print("Loaded sub-model is a generic nn.Module. Creating model from CFG and loading state_dict.")
                 model_from_cfg = YOLOModel(cfg=args.cfg, ch=3, nc=None).to(device) # nc, ch from common defaults
                 model_from_cfg.load_state_dict(loaded_sub_model.state_dict())
                 final_model = model_from_cfg.eval()

        elif isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'model_state_dict' in checkpoint):
            state_dict_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
            print(f"Checkpoint is a dictionary with '{state_dict_key}'. Creating model from CFG.")
            state_dict = checkpoint[state_dict_key]
            model_from_cfg = YOLOModel(cfg=args.cfg, ch=3, nc=None).to(device)
            model_from_cfg.load_state_dict(state_dict)
            final_model = model_from_cfg.eval()
        else: # Assume checkpoint is a raw state_dict
            print("Checkpoint is assumed to be a raw state_dict. Creating model from CFG.")
            model_from_cfg = YOLOModel(cfg=args.cfg, ch=3, nc=None).to(device)
            model_from_cfg.load_state_dict(checkpoint)
            final_model = model_from_cfg.eval()

        if final_model is None:
            raise ValueError("Could not load model from checkpoint. Ensure format is compatible.")

        model = final_model # Assign to the script's 'model' variable

        if hasattr(model, 'stride'):
            model_stride = int(model.stride.max() if isinstance(model.stride, torch.Tensor) else model.stride)
        print(f"PyTorch model loaded successfully on device {args.device}. Using stride: {model_stride}")

    except ImportError: # For Model or LoadImages
        print("Exiting due to Model or LoadImages class import failure.", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error loading PyTorch model. Please check paths, file format, and CFG compatibility: {e}", file=sys.stderr)
        return

    # Data Loading
    all_processed_images = []
    if args.dataset_path:
        print(f"Loading dataset from: {args.dataset_path}")
        try:
            # auto=False for letterbox padding without auto shape adjustment
            dataset_loader = LoadImages(args.dataset_path, img_size=args.img_size, stride=model_stride, auto=False)
            for path, img_processed, img0, vid_cap, s_shape in dataset_loader:
                # img_processed from LoadImages is CHW, RGB, uint8 (this is the expected format from utils.datasets.LoadImages)
                img_tensor = torch.from_numpy(img_processed).to(device)
                img_tensor = img_tensor.float() / 255.0  # Normalize to [0.0, 1.0]
                all_processed_images.append(img_tensor)

            if not all_processed_images:
                print(f"Error: No images found or loaded from dataset path: {args.dataset_path}", file=sys.stderr)
                return
            print(f"Loaded {len(all_processed_images)} images from dataset.")
        except Exception as e:
            print(f"Error loading dataset: {e}", file=sys.stderr)
            return
    else:
        print("Using dummy data for benchmarking.")
        # Create one batch of dummy data and reuse it
        dummy_input_batch = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(device)

    # Batch provider for real data
    current_image_idx = 0
    def get_real_batch_from_preloaded():
        nonlocal current_image_idx
        batch_frames = []
        for _ in range(args.batch_size):
            img_tensor = all_processed_images[current_image_idx % len(all_processed_images)]
            batch_frames.append(img_tensor)
            current_image_idx += 1
        return torch.stack(batch_frames)

    # Warm-up Phase
    try:
        print(f"\nStarting {args.warmup_runs} warm-up runs...")
        for _ in range(args.warmup_runs):
            with torch.no_grad():
                if args.dataset_path:
                    batch_data = get_real_batch_from_preloaded()
                    _ = model(batch_data)
                else:
                    _ = model(dummy_input_batch)
        print("Warm-up runs completed.")
    except Exception as e:
        print(f"Error during warm-up runs: {e}", file=sys.stderr)
        return

    # Benchmark Phase
    try:
        print(f"\nStarting {args.benchmark_runs} benchmark runs...")
        timings = []
        for i in range(args.benchmark_runs):
            start_time = time.perf_counter()
            with torch.no_grad():
                if args.dataset_path:
                    batch_data = get_real_batch_from_preloaded()
                    _ = model(batch_data)
                else:
                    _ = model(dummy_input_batch)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{args.benchmark_runs} benchmark runs...")
        print("Benchmark runs completed.")
    except Exception as e:
        print(f"Error during benchmark runs: {e}", file=sys.stderr)
        return

    # Results Calculation & Output
    if not timings:
        print("No benchmark timings recorded.", file=sys.stderr)
        return

    avg_time_s = sum(timings) / len(timings)
    avg_time_ms = avg_time_s * 1000
    fps = 1 / avg_time_s if avg_time_s > 0 else 0

    print("\n--- Native PyTorch Benchmark Results ---")
    print(f"Model: {args.weights}")
    print(f"Config: {args.cfg}")
    print(f"Dataset: {args.dataset_path if args.dataset_path else 'Dummy Data'}")
    print(f"Image Size (model input): {args.img_size}x{args.img_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {device.type}")
    print(f"Number of warm-up runs: {args.warmup_runs}")
    print(f"Number of benchmark runs: {args.benchmark_runs}")
    print("--------------------------------------")
    print(f"Average inference time: {avg_time_ms:.2f} ms")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print("--------------------------------------\n")

if __name__ == '__main__':
    main()
