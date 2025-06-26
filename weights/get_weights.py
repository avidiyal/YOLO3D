# This script is intended to be run from the 'weights/' directory.
# Example usage from project root:
# cd weights
# python get_weights.py  # Downloads resnet18.pkl and yolov5s.pt
# python get_weights.py --regressor_model resnet18
# python get_weights.py --yolo_model yolov5m
# python get_weights.py --regressor_model vgg11 --yolo_model yolov5l

import argparse
import gdown
import requests
import os
from pathlib import Path
import sys

# Constants for YOLO Download
STABLE_YOLO_RELEASE_TAG = "v7.0"
YOLO_DOWNLOAD_URL_PATTERN = "https://github.com/ultralytics/yolov5/releases/download/{tag}/{filename}"

# Dictionary for regressor model Google Drive IDs
PKL_WEIGHTS_GDRIVE_IDS = {
    'resnet': '1Bw4gUsRBxy8XZDGchPJ_URQjbHItikjw',
    'resnet18': '1k_v1RrDO6da_NDhBtMZL5c0QSogCmiRn',
    'vgg11': '1vZcB-NaPUCovVA-pH-g-3NNJuUA948ni'
}

def download_regressor_model(pkl_name):
    """Downloads a regressor model from Google Drive using its name."""
    if pkl_name not in PKL_WEIGHTS_GDRIVE_IDS:
        print(f"Warning: regressor model name '{pkl_name}' not recognized. Available: {list(PKL_WEIGHTS_GDRIVE_IDS.keys())}", file=sys.stderr)
        return

    print(f"Downloading {pkl_name}.pkl from Google Drive...")
    file_id = PKL_WEIGHTS_GDRIVE_IDS[pkl_name]
    url = f"https://drive.google.com/uc?id={file_id}"
    output_filename = f"{pkl_name}.pkl"

    gdown.download(url, output_filename, quiet=False)

def download_yolo_model(yolo_model_name_stem):
    """Downloads a YOLOv5 .pt model from GitHub releases."""
    model_filename = yolo_model_name_stem + '.pt'
    download_url = YOLO_DOWNLOAD_URL_PATTERN.format(tag=STABLE_YOLO_RELEASE_TAG, filename=model_filename)
    output_path = Path(model_filename)

    print(f"Downloading {model_filename} from {download_url}...")
    try:
        response = requests.get(download_url, timeout=120) 
        response.raise_for_status()
        output_path.write_bytes(response.content)

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"Successfully saved {model_filename} to {output_path.resolve()}")
        else:
            print(f"Error: {model_filename} saved but is empty or missing.", file=sys.stderr)
            if output_path.exists(): os.remove(output_path) # Clean up
    except Exception as e:
        print(f"Failed to download or save {model_filename}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained model weights. "
                    "Run from the 'weights/' directory. If no arguments are provided, "
                    "downloads default ResNet18 (PKL) and YOLOv5s (.pt)."
    )
    parser.add_argument(
        '--regressor_model', type=str, default='resnet18',
        help=(f"Name of the regressor model to download (e.g., 'resnet18', 'vgg11'). "
              f"Available: {list(PKL_WEIGHTS_GDRIVE_IDS.keys())}. Default: resnet18")
    )
    parser.add_argument(
        '--yolo_model', type=str, default='yolov5s',
        help="Name of the YOLOv5 .pt model to download (e.g., 'yolov5s', 'yolov5m'). '.pt' will be appended. Default: yolov5s"
    )
    args = parser.parse_args()

    # Download models
    download_regressor_model(args.regressor_model)
    print("-" * 30)
    download_yolo_model(args.yolo_model)
    
    print(f"\nDownloaded {args.regressor_model}, {args.yolo_model} .")

if __name__ == '__main__':
    main()
