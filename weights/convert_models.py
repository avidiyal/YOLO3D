# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
from pathlib import Path

# --- Add project root to sys.path ---
FILE_ROOT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = FILE_ROOT_SCRIPT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    print(f"Added {PROJECT_ROOT} to sys.path")

# --- Imports ---
import torch
import numpy as np
import openvino as ov

def load_pytorch_model(weights_path, cfg_path):
    """Load PyTorch YOLO model."""
    from models.yolo import Model as YOLOModel
    
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    model = YOLOModel(cfg=cfg_path, ch=3, nc=None).cpu()
    
    model.load_state_dict(checkpoint['model'].state_dict())
    
    return model.eval()

def load_pkl_model(pkl_path):
    """Load PKL ResNet18 model."""
    from script.Model import ResNet18
    from torchvision.models import resnet18 as torchvision_resnet18
    
    base_model = torchvision_resnet18(weights=None, progress=False)
    model = ResNet18(model=base_model)
    checkpoint = torch.load(pkl_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])

    return model.eval().cpu()

def convert_to_onnx(model, output_path, input_shape, input_names, output_names, dynamic_axes, opset_version):
    """Convert PyTorch model to ONNX."""
    dummy_input = torch.randn(*input_shape, device='cpu')
    
    torch.onnx.export(
        model, dummy_input, str(output_path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print(f"ONNX saved: {output_path}")

def convert_to_ir(onnx_path, ir_path):
    """Convert ONNX to OpenVINO IR."""
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(ir_path), compress_to_fp16=False)
    print(f"IR saved: {ir_path}")

def convert_pytorch_to_ir(weights_path, cfg_path, onnx_dir, ir_dir, img_size, batch_size, opset_version):
    """Convert PyTorch model to IR."""
    print(f"\n--- Converting PyTorch Model ---")
    print(f"Weights: {weights_path}")
    
    model = load_pytorch_model(weights_path, cfg_path)
    
    model_name = Path(weights_path).stem
    onnx_path = Path(onnx_dir) / f"{model_name}.onnx"
    ir_path = Path(ir_dir) / f"{model_name}.xml"
    
    # Convert to ONNX
    input_shape = (batch_size, 3, *img_size)
    convert_to_onnx(
        model, onnx_path, input_shape,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={'images': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=opset_version
    )
    
    # Convert to IR
    convert_to_ir(onnx_path, ir_path)

def convert_pkl_to_ir(pkl_path, onnx_dir, ir_dir, img_size, batch_size, opset_version):
    """Convert PKL model to IR."""
    print(f"\n--- Converting Regressor Model ---")
    print(f"PKL: {pkl_path}")
    
    model = load_pkl_model(pkl_path)
    
    model_name = Path(pkl_path).stem
    onnx_path = Path(onnx_dir) / f"{model_name}.onnx"
    ir_path = Path(ir_dir) / f"{model_name}.xml"
    
    # Convert to ONNX
    input_shape = (batch_size, 3, *img_size)
    convert_to_onnx(
        model, onnx_path, input_shape,
        input_names=['input'],
        output_names=['orient', 'conf', 'dim'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'orient': {0: 'batch_size'},
            'conf': {0: 'batch_size'},
            'dim': {0: 'batch_size'}
        },
        opset_version=opset_version
    )
    
    # Convert to IR
    convert_to_ir(onnx_path, ir_path)

def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch (.pt) or PKL (.pkl) models to OpenVINO IR format."
    )
    
    # Model paths
    parser.add_argument('--pytorch_weights', type=str, default=str(PROJECT_ROOT / "weights/yolov5s.pt"), help="PyTorch .pt weights file")
    parser.add_argument('--pytorch_cfg', type=str, default=str(PROJECT_ROOT / "models/yolov5s.yaml"), help="PyTorch .yaml config file")
    parser.add_argument('--pkl_path', type=str, default=str(PROJECT_ROOT / "weights/resnet18.pkl"), help="PKL model file")
    
    # Output directories
    parser.add_argument('--onnx_dir', type=str, default=str(PROJECT_ROOT / "weights/onnx"))
    parser.add_argument('--ir_dir', type=str, default=str(PROJECT_ROOT / "weights/openvino"))
    
    # PyTorch parameters
    parser.add_argument('--img_size_pytorch', nargs=2, type=int, default=[640, 640])
    parser.add_argument('--batch_size_pytorch', type=int, default=1)
    parser.add_argument('--opset_version_pytorch', type=int, default=12)
    
    # PKL parameters
    parser.add_argument('--img_size_pkl', nargs=2, type=int, default=[224, 224])
    parser.add_argument('--batch_size_pkl', type=int, default=1)
    parser.add_argument('--opset_version_pkl', type=int, default=11)
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.onnx_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ir_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert models if files exist
    if Path(args.pytorch_weights).exists() and Path(args.pytorch_cfg).exists():
        convert_pytorch_to_ir(
            args.pytorch_weights, args.pytorch_cfg,
            args.onnx_dir, args.ir_dir,
            args.img_size_pytorch, args.batch_size_pytorch, args.opset_version_pytorch
        )
    
    if Path(args.pkl_path).exists():
        convert_pkl_to_ir(
            args.pkl_path,
            args.onnx_dir, args.ir_dir, 
            args.img_size_pkl, args.batch_size_pkl, args.opset_version_pkl
        )
    
    print("\nConversion complete!")

if __name__ == '__main__':
    main()