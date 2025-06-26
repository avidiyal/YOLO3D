# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
"""

import argparse
import os
import sys
from pathlib import Path
import glob

import cv2
import torch
import numpy as np
from collections import deque
from torchvision.models import resnet18, vgg11

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import LOGGER, check_img_size, check_requirements, non_max_suppression, print_args, scale_coords
from utils.torch_utils import select_device, time_sync

from script.Dataset import generate_bins, DetectedObject
from library.Math import *
from library.Plotting import *
from script import Model, ClassAverages
from script.Model import ResNet, ResNet18, VGG11

# Model factory to choose model
model_factory = {
    'resnet': resnet18(pretrained=True),
    'resnet18': resnet18(pretrained=True),
    # 'vgg11': vgg11(pretrained=True)
}
regressor_factory = {
    'resnet': ResNet,
    'resnet18': ResNet18,
    'vgg11': VGG11
}

class Bbox:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_

tracking_trajectories = {}

def detect3d(
    reg_weights,
    model_select,
    source,
    calib_file,
    show_result,
    save_result,
    output_path
    ):

    # Directory or video file
    imgs_path = [source] if source.endswith(('.mp4', '.avi', '.mov')) else sorted(glob.glob(str(source) + '/*'))
    calib = str(calib_file)

    # Load model
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model).to('cpu')

    # Load weight
    checkpoint = torch.load(reg_weights, map_location='cpu')
    regressor.load_state_dict(checkpoint['model_state_dict'])
    regressor.eval()

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    # Initialize VideoWriter if saving result
    if save_result and output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        video_path = os.path.join(output_path, 'output_video_1122west.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = None

    # Loop images or video frames
    for i, img_path in enumerate(imgs_path):
        # Read image or video frame
        if img_path.endswith(('.mp4', '.avi', '.mov')):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Video file not found: {img_path}")

            cap = cv2.VideoCapture(img_path)
            if not cap.isOpened():
                raise IOError(f"Error opening video file: {img_path}")

            frame_index = 0
            if save_result and out is None:
                # Get video properties
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
                if not out.isOpened():
                    raise IOError(f"Error opening video writer for file: {video_path}")

            while cap.isOpened():
                ret, img = cap.read()
                if not ret:
                    break
                process_frame(img, regressor, averages, angle_bins, calib, show_result, save_result, out, frame_index)
                frame_index += 1
            cap.release()
        else:
            img = cv2.imread(img_path)
            process_frame(img, regressor, averages, angle_bins, calib, show_result, save_result, out, i)

    # Release VideoWriter
    if save_result and out is not None:
        out.release()

def process_frame(img, regressor, averages, angle_bins, calib, show_result, save_result, out, frame_index):
    # Run detection 2d
    img2D, bboxes2d = process2D(img, track=True)

    for det in bboxes2d:
        if not averages.recognized_class(det.detected_class):
            continue
        try: 
            detectedObject = DetectedObject(img, det.detected_class, det.box_2d, calib)
        except:
            continue

        theta_ray = detectedObject.theta_ray
        input_img = detectedObject.img
        proj_matrix = detectedObject.proj_matrix
        box_2d = det.box_2d
        detected_class = det.detected_class

        input_tensor = torch.zeros([1,3,224,224]).to('cpu')
        input_tensor[0,:,:,:] = input_img

        # Predict orient, conf, and dim
        [orient, conf, dim] = regressor(input_tensor)
        orient = orient.cpu().data.numpy()[0, :, :]
        conf = conf.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]

        dim += averages.get_item(detected_class)

        argmax = np.argmax(conf)
        orient = orient[argmax, :]
        cos = orient[0]
        sin = orient[1]
        alpha = np.arctan2(sin, cos)
        alpha += angle_bins[argmax]
        alpha -= np.pi

        # Plot 3d detection
        plot3d(img, proj_matrix, box_2d, dim, alpha, theta_ray)

    if show_result:
        cv2.imshow('3d detection', img)
        cv2.waitKey(1)  # Use waitKey(1) for video to update frames

    if save_result and out is not None:
        out.write(img)  # Write frame to video

@torch.no_grad()
def process2D(image, track=True, device='cpu'):
    bboxes = []
    # Load model
    weights = '../weights/yolov5s.pt'
    data = '../data/coco128.yaml'
    imgsz = [640, 640]
    classes = [0, 2, 3, 5]

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    im0s = image
    im = cv2.resize(im0s, (imgsz[1], imgsz[0]))
    im = im.transpose(2, 0, 1)  # HWC to CHW
    im = np.ascontiguousarray(im)

    # Create a single-item dataset
    dataset = [(None, im, im0s, None, '')]

    model = torch.compile(model, backend="openvino", options={"device": "CPU"})
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(prediction=pred, classes=classes)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p) if p else ''  # to Path if not None
            s += '%gx%g ' % im.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy_ = (torch.tensor(xyxy).view(1,4)).view(-1).tolist()
                    xyxy_ = [int(x) for x in xyxy_]
                    top_left, bottom_right = (xyxy_[0], xyxy_[1]), (xyxy_[2], xyxy_[3])
                    bbox = [top_left, bottom_right]
                    c = int(cls)  # integer class
                    label = names[c]

                    bboxes.append(Bbox(bbox, label))

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    return im0s, bboxes

def plot3d(
    img,
    proj_matrix,
    box_2d,
    dimensions,
    alpha,
    theta_ray,
    img_2d=None
    ):

    # The math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, proj_matrix, orient, dimensions, location) # 3d boxes

    return location

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../weights/yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='../eval/image_2', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='../data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', default=[0, 2, 3, 5], nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--reg_weights', type=str, default='../weights/resnet18.pkl', help='Regressor model weights')
    parser.add_argument('--model_select', type=str, default='resnet', help='Regressor model list: resnet, vgg, eff')
    parser.add_argument('--calib_file', type=str, default='../eval/camera_cal/calib_cam_to_cam.txt', help='Calibration file or path')
    parser.add_argument('--show_result', action='store_false', help='Show Results with imshow')
    parser.add_argument('--save_result', action='store_true', help='Save result')
    parser.add_argument('--output_path', type=str, default='../output', help='Save output path')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    detect3d(
        reg_weights=opt.reg_weights,
        model_select=opt.model_select,
        source=opt.source,
        calib_file=opt.calib_file,
        show_result=opt.show_result,
        save_result=opt.save_result,
        output_path=opt.output_path
    )

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
