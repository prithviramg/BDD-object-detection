import os
import json
import math
import argparse
import numpy as np
import torch
import tqdm
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Import the model definition
from model import ObjectDetectionModel

# --------------------
# Utility functions
# --------------------

def iou(boxA, boxB):
    """Computes Intersection over Union (IoU) between two boxes.
       Box format: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou_value = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou_value

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def pool_nms(heat, kernel=3):
    """Simple NMS on heatmaps using max pooling."""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def decode_detections(heatmap, wh, offset, down_ratio, threshold=0.3, max_detections=100):
    """
    Decodes predictions from one branch.
      heatmap: tensor [num_classes, H, W]
      wh: tensor [2, H, W] (predicted width and height)
      offset: tensor [2, H, W] (predicted offset)
    Returns a list of detections:
      Each detection is a dict: { 'cls': class_id, 'score': score, 'bbox': [x1, y1, x2, y2] }
    """
    detections = []
    # Apply NMS on heatmap
    heatmap = pool_nms(heatmap)
    num_classes, H, W = heatmap.shape
    heat_np = heatmap.cpu().numpy()
    wh_np = wh.cpu().numpy()
    offset_np = offset.cpu().numpy()

    for cls in range(num_classes):
        cls_heat = heat_np[cls]
        ys, xs = np.where(cls_heat > threshold)
        for y, x in zip(ys, xs):
            score = cls_heat[y, x]
            off = offset_np[:, y, x]
            wh_val = wh_np[:, y, x]
            # Convert feature map coordinates (with offset) back to image space.
            cx = (x + off[0]) * down_ratio
            cy = (y + off[1]) * down_ratio
            w_box = wh_val[0] * down_ratio * W
            h_box = wh_val[1] * down_ratio * H
            x1 = cx - w_box / 2
            y1 = cy - h_box / 2
            x2 = cx + w_box / 2
            y2 = cy + h_box / 2
            detections.append({
                'cls': cls,
                'score': score,
                'bbox': [x1, y1, x2, y2]
            })
    # Keep top detections sorted by score.
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)[:max_detections]
    return detections

def load_image(image_path, input_size):
    """Loads image with OpenCV, converts to RGB, resizes to input_size, and returns both resized and original image."""
    orig = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(orig)
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
    ])
    img_tensor = transform(pil_img)
    return img_tensor.unsqueeze(0), orig

# --------------------
# Metric calculation functions
# --------------------

def compute_ap(recalls, precisions):
    """Computes Average Precision (AP) using the interpolation method."""
    # Append sentinel values at the start and end.
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    # Make the precision monotonically decreasing.
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    # Identify points where recall changes.
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def evaluate_detections(preds, gts, iou_threshold=0.5):
    """
    Evaluates predictions against ground truth boxes for a single class.
    preds: list of predictions for the class.
           Each prediction is a dict: { 'image_id': str, 'score': float, 'bbox': [x1, y1, x2, y2] }
    gts: dict mapping image_id to list of ground truth boxes (each is [x1, y1, x2, y2]).
    Returns: precision, recall arrays and AP.
    """
    # Count total ground truths.
    npos = sum([len(gts[img_id]) for img_id in gts])
    
    # Sort predictions by score in descending order.
    preds = sorted(preds, key=lambda x: x['score'], reverse=True)
    TP = np.zeros(len(preds))
    FP = np.zeros(len(preds))
    
    # For each ground truth box, keep track if it is matched.
    gt_detected = {img_id: np.zeros(len(gts[img_id])) for img_id in gts}
    
    for i, pred in enumerate(preds):
        img_id = pred['image_id']
        pred_box = pred['bbox']
        max_iou = 0.0
        max_idx = -1
        if img_id in gts:
            for j, gt_box in enumerate(gts[img_id]):
                iou_val = iou(pred_box, gt_box)
                if iou_val > max_iou:
                    max_iou = iou_val
                    max_idx = j
        if max_iou >= iou_threshold:
            if gt_detected[img_id][max_idx] == 0:
                TP[i] = 1  # True positive
                gt_detected[img_id][max_idx] = 1
            else:
                FP[i] = 1  # Duplicate detection
        else:
            FP[i] = 1

    # Compute cumulative true positives and false positives.
    cum_TP = np.cumsum(TP)
    cum_FP = np.cumsum(FP)
    
    recalls = cum_TP / (npos + 1e-6)
    precisions = cum_TP / (cum_TP + cum_FP + 1e-6)
    
    ap = compute_ap(recalls, precisions)
    return precisions, recalls, ap

# --------------------
# Main Evaluation Script
# --------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Object Detection Model on BDD Validation Dataset")
    parser.add_argument('--data_dir', type=str, default=r"D:\database\bdd_dataset\bdd100k_images_100k\bdd100k\images\100k\val", help="Path to BDD validation images directory")
    parser.add_argument('--anno_file', type=str, default=r"D:\database\bdd_dataset\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_val.json", help="Path to BDD validation JSON annotations")
    parser.add_argument('--checkpoint', type=str, default=r"D:\GIT\BDD-object-detection\checkpoints\model_epoch_1.pth", help="Path to model checkpoint (.pth)")
    parser.add_argument('--input_size', type=int, nargs=2, default=[720, 1280], help="Input size (H, W)")
    parser.add_argument('--tc_threshold', type=float, default=0.1, help="Detection threshold for TC branch")
    parser.add_argument('--tp_threshold', type=float, default=0.1, help="Detection threshold for TP branch")
    parser.add_argument('--iou_threshold', type=float, default=0.5, help="IoU threshold for matching")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Evaluating on device:", device)

    # Define class names and mappings.
    tc_class_names = ['traffic light', 'traffic sign']
    tp_class_names = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motor', 'bike']
    # For evaluation, we create a unified mapping. Use prefix to distinguish if desired.
    all_class_names = tc_class_names + tp_class_names
    # Create mapping: class_id -> class name.
    # TC classes: 0,1 ; TP classes: 2-9.
    
    # Load model.
    model = ObjectDetectionModel(tp_class=8, tc_class=2, bifpn_channels=64)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load annotations JSON.
    with open(args.anno_file, 'r') as f:
        annotations = json.load(f)

    # Build ground truth dictionary:
    # Mapping image_id -> { 'boxes': list of bboxes, 'labels': list of class indices }
    gt_dict = {}
    for ann in tqdm.tqdm(annotations[:200]):
        image_id = ann['name']
        # Open the image to get original dimensions.
        img_path = os.path.join(args.data_dir, image_id)
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            continue
        orig_h, orig_w = img_cv.shape[0:2]
        scale_x = args.input_size[1] / orig_w
        scale_y = args.input_size[0] / orig_h

        gt_boxes = []
        gt_labels = []
        for obj in ann['labels']:
            category = obj['category']
            if 'box2d' not in obj:
                continue
            box = obj['box2d']
            x1 = box['x1'] * scale_x
            y1 = box['y1'] * scale_y
            x2 = box['x2'] * scale_x
            y2 = box['y2'] * scale_y
            # Determine unified class index.
            if category in tc_class_names:
                cls_idx = tc_class_names.index(category)
            elif category in tp_class_names:
                cls_idx = len(tc_class_names) + tp_class_names.index(category)
            else:
                continue
            gt_boxes.append([x1, y1, x2, y2])
            gt_labels.append(cls_idx)
        if len(gt_boxes) > 0:
            gt_dict[image_id] = {'boxes': gt_boxes, 'labels': gt_labels}

    # Prepare lists to hold predictions for each class.
    pred_dict = {cls_idx: [] for cls_idx in range(len(all_class_names))}

    # Inference parameters per branch.
    tc_down = 2  # TC branch output resolution factor
    tp_down = 4  # TP branch output resolution factor

    transform = transforms.Compose([
        transforms.Resize(tuple(args.input_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
    ])

    # Iterate over the validation annotations.
    for ann in tqdm.tqdm(annotations[:200]):
        image_id = ann['name']
        img_path = os.path.join(args.data_dir, image_id)
        img_tensor, orig_img = load_image(img_path, tuple(args.input_size))
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
        # Decode detections from both branches.
        tc_heat = outputs["tc"]["heatmap"].squeeze(0)    # shape: [num_tc, H, W]
        tc_wh   = outputs["tc"]["size"].squeeze(0)         # shape: [2, H, W]
        tc_off  = outputs["tc"]["offset"].squeeze(0)       # shape: [2, H, W]
        tp_heat = outputs["tp"]["heatmap"].squeeze(0)       # shape: [num_tp, H, W]
        tp_wh   = outputs["tp"]["size"].squeeze(0)
        tp_off  = outputs["tp"]["offset"].squeeze(0)

        tc_dets = decode_detections(tc_heat, tc_wh, tc_off, down_ratio=tc_down, threshold=args.tc_threshold)
        tp_dets = decode_detections(tp_heat, tp_wh, tp_off, down_ratio=tp_down, threshold=args.tp_threshold)
        
        # Adjust class indices: TC detections remain as 0 and 1; TP detections need to be shifted by len(tc_class_names)
        for det in tc_dets:
            det['image_id'] = image_id
            # cls already in [0,1] for TC.
            pred_dict[det['cls']].append(det)
        for det in tp_dets:
            det['image_id'] = image_id
            # shift TP class indices by len(tc_class_names)
            det['cls'] = len(tc_class_names) + det['cls']
            pred_dict[det['cls']].append(det)
    
    # Now, evaluate per-class AP.
    aps = {}
    for cls_idx, cls_name in enumerate(all_class_names):
        # Collect all predictions and ground truths for this class.
        preds_cls = pred_dict[cls_idx]
        # Build ground truth for this class.
        gt_cls = {}
        for image_id, item in gt_dict.items():
            boxes = []
            for label, box in zip(item['labels'], item['boxes']):
                if label == cls_idx:
                    boxes.append(box)
            if boxes:
                gt_cls[image_id] = boxes
        if len(gt_cls) == 0:
            print(f"No ground truth for class '{cls_name}'")
            aps[cls_name] = 0.0
            continue
        precisions, recalls, ap = evaluate_detections(preds_cls, gt_cls, iou_threshold=args.iou_threshold)
        aps[cls_name] = ap
        print(f"Class: {cls_name:15s} | AP: {ap:.4f}")
        
        # Optionally, plot precision-recall curve for each class.
        plt.figure()
        plt.step(recalls, precisions, where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall curve for {cls_name} (AP: {ap:.4f})')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.grid(True)
        plt.savefig(f'PR_{cls_name}.png')
        plt.close()

    # Compute mean AP.
    mAP = np.mean(list(aps.values()))
    print("\nOverall mAP: {:.4f}".format(mAP))

if __name__ == "__main__":
    main()
