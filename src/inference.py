import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Import your model definition
from model import ObjectDetectionModel

def load_model(checkpoint_path, device):
    # Instantiate the model
    model = ObjectDetectionModel(tp_class=8, tc_class=2, bifpn_channels=64)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, input_size):
    """
    Reads an image from path, resizes to input_size (H, W) and converts to tensor.
    Also returns the original image (as a NumPy array) for visualization.
    """
    orig_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(orig_image)
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
    ])
    image_tensor = transform(pil_img)
    return image_tensor.unsqueeze(0), orig_image

def pool_nms(heat, kernel=3):
    # Apply max pooling to heatmap to extract local peaks.
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def decode_detections(heatmap, wh, offset, down_ratio, threshold=0.1, max_detections=100):
    """
    Decodes a single branch of predictions.
      heatmap: tensor of shape [num_classes, H, W]
      wh: tensor of shape [2, H, W] (predicted width and height)
      offset: tensor of shape [2, H, W] (predicted offsets)
      down_ratio: factor to convert feature map coordinates back to image coordinates.
    Returns a list of detections where each detection is a dictionary:
      { 'cls': class_id, 'score': score, 'bbox': [x1, y1, x2, y2] }
    """
    detections = []
    # Apply NMS by pooling and then threshold
    heatmap = pool_nms(heatmap)
    
    num_classes, H, W = heatmap.shape
    heat_np = heatmap.cpu().numpy()
    wh_np = wh.cpu().numpy()
    offset_np = offset.cpu().numpy()

    for cls in range(num_classes):
        cls_heat = heat_np[cls]
        # Get indices where probability is above threshold
        ys, xs = np.where(cls_heat > threshold)
        for y, x in zip(ys, xs):
            score = cls_heat[y, x]
            # Get local offset and size
            off = offset_np[:, y, x]
            wh_val = wh_np[:, y, x]
            # Compute center in feature map coordinate and then to image coordinate
            cx = (x + off[0]) * down_ratio
            cy = (y + off[1]) * down_ratio
            w_box, h_box = wh_val[0] * down_ratio * W, wh_val[1] * down_ratio * H
            # Convert to top-left & bottom-right
            x1 = cx - w_box / 2
            y1 = cy - h_box / 2
            x2 = cx + w_box / 2
            y2 = cy + h_box / 2
            detections.append({
                'cls': cls,
                'score': score,
                'bbox': [x1, y1, x2, y2]
            })
    # Optionally sort and keep top max_detections
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)[:max_detections]
    return detections

def draw_detections(image, detections, class_names, color):
    """
    Draw bounding boxes on the image.
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        cls_id = det['cls']
        label = f"{class_names[cls_id]}: {score:.2f}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, thickness=1)
    return image

def main():
    parser = argparse.ArgumentParser(description="Inference for Object Detection Model")
    parser.add_argument('--image', type=str, default=r"D:\database\bdd_dataset\bdd100k_images_100k\bdd100k\images\100k\test\cabf7be1-36a39a28.jpg", help="Path to input image")
    parser.add_argument('--checkpoint', type=str, default=r"D:\GIT\BDD-object-detection\checkpoints\model_iter_33500.pth", help="Path to model checkpoint (.pth)")
    parser.add_argument('--input_size', type=int, nargs=2, default=[720, 1280], help="Input size (H, W)")
    parser.add_argument('--tc_threshold', type=float, default=0.15, help="Heatmap threshold for TC branch")
    parser.add_argument('--tp_threshold', type=float, default=0.15, help="Heatmap threshold for TP branch")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load model
    model = load_model(args.checkpoint, device)

    # Load image and preprocess
    image_tensor, orig_image = preprocess_image(args.image, tuple(args.input_size))
    image_tensor = image_tensor.to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)

    # The model outputs dictionaries for 'tc' and 'tp'
    # For inference decoding we use:
    #   tc: feature map resolution is input_size / 2
    #   tp: feature map resolution is input_size / 4
    tc_down = 2
    tp_down = 4

    # Squeeze batch dimension and get outputs
    tc_heatmap = outputs["tc"]["heatmap"].squeeze(0)  # shape: [num_classes, H, W]
    tc_wh = outputs["tc"]["size"].squeeze(0)            # shape: [2, H, W]
    tc_offset = outputs["tc"]["offset"].squeeze(0)        # shape: [2, H, W]

    tp_heatmap = outputs["tp"]["heatmap"].squeeze(0)
    tp_wh = outputs["tp"]["size"].squeeze(0)
    tp_offset = outputs["tp"]["offset"].squeeze(0)

    # Decode detections from both branches
    # (Define class names for visualization.)
    tc_class_names = ['traffic light', 'traffic sign']
    tp_class_names = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motor', 'bike']

    tc_dets = decode_detections(tc_heatmap, tc_wh, tc_offset, down_ratio=tc_down, threshold=args.tc_threshold)
    tp_dets = decode_detections(tp_heatmap, tp_wh, tp_offset, down_ratio=tp_down, threshold=args.tp_threshold)

    # Convert original image to BGR for OpenCV if needed, here we work in RGB.
    vis_image = orig_image.copy()

    # Draw detections from each branch with different colors.
    vis_image = draw_detections(vis_image, tc_dets, tc_class_names, color=(255, 0, 0))
    vis_image = draw_detections(vis_image, tp_dets, tp_class_names, color=(0, 255, 0))

    # Plot results using matplotlib.
    plt.figure(figsize=(12, 8))
    plt.imshow(vis_image)
    plt.title("Inference Results")
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    main()
