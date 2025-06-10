# import torch

# # Load the model from pkl
# checkpoint = torch.load('resnet18.pkl', map_location='cpu', weights_only=False)

# # If checkpoint is already the state_dict, save it directly
# torch.save(checkpoint, 'resnet18_nosd.pt')


import torch
import torch.onnx
import sys
from pathlib import Path

# Add parent directory to path so we can import custom modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# Import your ResNet18 class and needed modules
from script.Model import ResNet18  
from torchvision.models import resnet18

def convert_pkl_to_onnx(pkl_path='../resnet18.pkl', onnx_path='resnet18.onnx'):
    # Load the checkpoint
    checkpoint = torch.load(pkl_path, map_location='cpu', weights_only=False)
    
    print(f"Loaded checkpoint from {pkl_path}")
    
    # Create the model instance
    base_model = resnet18(pretrained=False)
    model = ResNet18(model=base_model).to('cpu')
    
    # Load state dict, using strict=False to handle mismatches
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Model weights loaded with strict=False")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    # Set to evaluation mode
    model.eval()
    
    # Create dummy input (adjust size if needed)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['orient', 'conf', 'dim'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'orient': {0: 'batch_size'},
            'conf': {0: 'batch_size'},
            'dim': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    convert_pkl_to_onnx()