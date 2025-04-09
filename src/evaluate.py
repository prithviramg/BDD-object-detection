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

from model import ObjectDetectionModel

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


def pool_nms(heat, kernel=3):
    """Simple NMS on heatmaps using max pooling."""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def decode_detections(
    heatmap, wh, offset, down_ratio, threshold=0.3, max_detections=100
):
    """
    Decodes predictions from one branch.
      heatmap: tensor [num_classes, H, W]
      wh: tensor [2, H, W] (predicted width and height)
      offset: tensor [2, H, W] (predicted offset)
    Returns a list of detections:
      Each detection is a dict: { 'cls': class_id, 'score': score, 'bbox': [x1, y1, x2, y2] }
    """
    detections = []
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
            cx = (x + off[0]) * down_ratio
            cy = (y + off[1]) * down_ratio
            w_box = wh_val[0] * down_ratio * W
            h_box = wh_val[1] * down_ratio * H
            x1 = cx - w_box / 2
            y1 = cy - h_box / 2
            x2 = cx + w_box / 2
            y2 = cy + h_box / 2
            detections.append({"cls": cls, "score": score, "bbox": [x1, y1, x2, y2]})
    detections = sorted(detections, key=lambda x: x["score"], reverse=True)[
        :max_detections
    ]
    return detections


def load_image(image_path, input_size):
    """Loads image with OpenCV, converts to RGB, resizes to input_size, and returns both resized and original image."""
    orig = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(orig)
    transform = transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    img_tensor = transform(pil_img)
    return img_tensor.unsqueeze(0), orig

def compute_ap(recalls, precisions):
    """Computes Average Precision (AP) using the interpolation method."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def evaluate_detections(preds, gts, iou_threshold=0.5):
    """
    Evaluates predictions against ground truth boxes for a single class.
    preds: list of predictions for the class.
           Each prediction is a dict: { 'image_id': str, 'score': float, 'bbox': [x1, y1, x2, y2] }
    gts: dict mapping image_id -> list of ground truth boxes.
    Returns:
      - precisions: precision array
      - recalls: recall array
      - ap: average precision
      - ar: average recall (final recall value)
      - avg_loc_error: average localization error (1 - IoU) for true positives
      - f1: F1 score (computed using final precision and recall)
    """
    npos = sum([len(gts[img_id]) for img_id in gts])
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    TP = np.zeros(len(preds))
    FP = np.zeros(len(preds))
    gt_detected = {img_id: np.zeros(len(gts[img_id])) for img_id in gts}
    loc_errors = []

    for i, pred in enumerate(preds):
        img_id = pred["image_id"]
        pred_box = pred["bbox"]
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
                TP[i] = 1
                gt_detected[img_id][max_idx] = 1
                loc_errors.append(1 - max_iou)
            else:
                FP[i] = 1
        else:
            FP[i] = 1

    cum_TP = np.cumsum(TP)
    cum_FP = np.cumsum(FP)
    recalls = cum_TP / (npos + 1e-6)
    precisions = cum_TP / (cum_TP + cum_FP + 1e-6)
    ap = compute_ap(recalls, precisions)
    ar = recalls[-1] if len(recalls) > 0 else 0.0
    avg_loc_error = np.mean(loc_errors) if loc_errors else 0.0
    final_precision = precisions[-1] if len(precisions) > 0 else 0.0
    final_recall = recalls[-1] if len(recalls) > 0 else 0.0
    if final_precision + final_recall > 0:
        f1 = 2 * final_precision * recalls[-1] / (final_precision + recalls[-1])
    else:
        f1 = 0.0
    return precisions, recalls, ap, ar, avg_loc_error, f1


def evaluate_subset_metrics(
    image_ids, pred_dict_all, gt_dict_all, all_class_names, iou_threshold
):
    """
    Compute metrics (mAP, AR, localization error, F1) over a subset of images.
    Filters predictions and ground truths to only include those in the provided image_ids.
    Returns a dict with keys: 'mAP', 'AR', 'loc_error', 'F1'
    """
    gt_subset = {}
    for img_id, item in gt_dict_all.items():
        if img_id in image_ids:
            gt_subset[img_id] = item[
                "boxes_labels"
            ]

    gt_by_class = {cls_idx: {} for cls_idx in range(len(all_class_names))}
    pred_by_class = {cls_idx: [] for cls_idx in range(len(all_class_names))}

    for img_id, item in gt_dict_all.items():
        if img_id in image_ids:
            boxes = item.get("boxes_labels", [])
            for box, label in boxes:
                if label in gt_by_class:
                    if img_id not in gt_by_class[label]:
                        gt_by_class[label][img_id] = []
                    gt_by_class[label][img_id].append(box)

    for cls_idx in range(len(all_class_names)):
        for pred in pred_dict_all[cls_idx]:
            if pred["image_id"] in image_ids:
                pred_by_class[cls_idx].append(pred)

    mAP_list = []
    AR_list = []
    loc_error_list = []
    F1_list = []
    for cls_idx in range(len(all_class_names)):
        preds_cls = pred_by_class[cls_idx]
        gt_cls = gt_by_class[cls_idx]
        if len(gt_cls) == 0:
            continue
        _, _, ap, ar, loc_err, f1 = evaluate_detections(
            preds_cls, gt_cls, iou_threshold
        )
        mAP_list.append(ap)
        AR_list.append(ar)
        loc_error_list.append(loc_err)
        F1_list.append(f1)
    # If a class has no ground truth, we skip it.
    mAP = np.mean(mAP_list) if mAP_list else 0.0
    AR = np.mean(AR_list) if AR_list else 0.0
    avg_loc_error = np.mean(loc_error_list) if loc_error_list else 0.0
    F1 = np.mean(F1_list) if F1_list else 0.0
    return {"mAP": mAP, "AR": AR, "loc_error": avg_loc_error, "F1": F1}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Object Detection Model on BDD Validation Dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\database\bdd_dataset\bdd100k_images_100k\bdd100k\images\100k\val",
        help="Path to BDD validation images directory",
    )
    parser.add_argument(
        "--anno_file",
        type=str,
        default=r"D:\database\bdd_dataset\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_val.json",
        help="Path to BDD validation JSON annotations",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"D:\GIT\BDD-object-detection\checkpoints\model_iter_26400.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--input_size", type=int, nargs=2, default=[720, 1280], help="Input size (H, W)"
    )
    parser.add_argument(
        "--tc_threshold",
        type=float,
        default=0.15,
        help="Detection threshold for TC branch",
    )
    parser.add_argument(
        "--tp_threshold",
        type=float,
        default=0.15,
        help="Detection threshold for TP branch",
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.5, help="IoU threshold for matching"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating on device:", device)

    # Define class names and mappings.
    tc_class_names = ["traffic light", "traffic sign"]
    tp_class_names = [
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motor",
        "bike",
    ]
    all_class_names = (
        tc_class_names + tp_class_names
    )  # TC: indices 0-1, TP: indices 2-9

    # Load model.
    model = ObjectDetectionModel(tp_class=8, tc_class=2, bifpn_channels=128)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load annotations JSON.
    with open(args.anno_file, "r") as f:
        annotations = json.load(f)

    # Build ground truth and metadata dictionaries.
    # We'll store for each image:
    #   gt_dict[image_id] = { 'boxes_labels': [(box, label), ...] }
    #   meta_dict[image_id] = { 'timeofday': str, 'weather': str }
    gt_dict = {}
    meta_dict = {}
    for ann in tqdm.tqdm(annotations[:100], desc="Building GT & meta"):
        image_id = ann["name"]
        img_path = os.path.join(args.data_dir, image_id)
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            continue
        orig_h, orig_w = img_cv.shape[0:2]
        scale_x = args.input_size[1] / orig_w
        scale_y = args.input_size[0] / orig_h

        boxes_labels = []
        for obj in ann["labels"]:
            category = obj["category"]
            if "box2d" not in obj:
                continue
            box = obj["box2d"]
            x1 = box["x1"] * scale_x
            y1 = box["y1"] * scale_y
            x2 = box["x2"] * scale_x
            y2 = box["y2"] * scale_y
            if category in tc_class_names:
                cls_idx = tc_class_names.index(category)
            elif category in tp_class_names:
                cls_idx = len(tc_class_names) + tp_class_names.index(category)
            else:
                continue
            boxes_labels.append(([x1, y1, x2, y2], cls_idx))
        if boxes_labels:
            gt_dict[image_id] = {"boxes_labels": boxes_labels}
            # Extract metadata if available (here, we assume they are stored in the "attributes" field).
            attributes = ann.get("attributes", {})
            timeofday = attributes.get("timeofday", "unknown")
            weather = attributes.get("weather", "unknown")
            meta_dict[image_id] = {"timeofday": timeofday, "weather": weather}

    # Prepare lists to hold predictions for each class.
    pred_dict = {cls_idx: [] for cls_idx in range(len(all_class_names))}

    # Inference parameters per branch.
    tc_down = 4  # TC branch output resolution factor
    tp_down = 8  # TP branch output resolution factor

    transform = transforms.Compose(
        [
            transforms.Resize(tuple(args.input_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
        ]
    )

    # Run inference and aggregate predictions.
    for ann in tqdm.tqdm(annotations[:100], desc="Running inference"):
        image_id = ann["name"]
        img_path = os.path.join(args.data_dir, image_id)
        try:
            img_tensor, orig_img = load_image(img_path, tuple(args.input_size))
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            continue
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
        # Decode detections from both branches.
        tc_heat = outputs["tc"]["heatmap"].squeeze(0)  # shape: [num_tc, H, W]
        tc_wh = outputs["tc"]["size"].squeeze(0)  # shape: [2, H, W]
        tc_off = outputs["tc"]["offset"].squeeze(0)  # shape: [2, H, W]
        tp_heat = outputs["tp"]["heatmap"].squeeze(0)  # shape: [num_tp, H, W]
        tp_wh = outputs["tp"]["size"].squeeze(0)
        tp_off = outputs["tp"]["offset"].squeeze(0)

        tc_dets = decode_detections(
            tc_heat, tc_wh, tc_off, down_ratio=tc_down, threshold=args.tc_threshold
        )
        tp_dets = decode_detections(
            tp_heat, tp_wh, tp_off, down_ratio=tp_down, threshold=args.tp_threshold
        )

        # Assign detections and adjust class indices.
        for det in tc_dets:
            det["image_id"] = image_id
            pred_dict[det["cls"]].append(det)
        for det in tp_dets:
            det["image_id"] = image_id
            det["cls"] = len(tc_class_names) + det["cls"]
            pred_dict[det["cls"]].append(det)

    # Now, evaluate per-class AP and additional metrics.
    aps = {}
    ars = {}
    loc_errors = {}
    f1_scores = {}
    for cls_idx, cls_name in enumerate(all_class_names):
        # Build ground truth for this class.
        gt_cls = {}
        for img_id, item in gt_dict.items():
            boxes = []
            for box, label in item["boxes_labels"]:
                if label == cls_idx:
                    boxes.append(box)
            if boxes:
                gt_cls[img_id] = boxes
        if len(gt_cls) == 0:
            print(f"No ground truth for class '{cls_name}'")
            aps[cls_name] = 0.0
            ars[cls_name] = 0.0
            loc_errors[cls_name] = 0.0
            f1_scores[cls_name] = 0.0
            continue
        precisions, recalls, ap, ar, loc_err, f1 = evaluate_detections(
            pred_dict[cls_idx], gt_cls, iou_threshold=args.iou_threshold
        )
        aps[cls_name] = ap
        ars[cls_name] = ar
        loc_errors[cls_name] = loc_err
        f1_scores[cls_name] = f1
        print(
            f"Class: {cls_name:15s} | AP: {ap:.4f} | AR: {ar:.4f} | LocError: {loc_err:.4f} | F1: {f1:.4f}"
        )

        # Plot precision-recall curve for each class.
        plt.figure()
        plt.step(recalls, precisions, where="post")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall for {cls_name} (AP: {ap:.4f})")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.grid(True)
        plt.savefig(f"PR_{cls_name}.png")
        # plt.show()
        plt.close()
    mAP_overall = np.mean(list(aps.values()))
    print("\nOverall mAP: {:.4f}".format(mAP_overall))

    # --------------------
    # Evaluation based on metadata (time-of-day and weather)
    # --------------------
    # Build subsets: keys -> list of image_ids
    timeofday_subsets = {}
    weather_subsets = {}
    for img_id, meta in meta_dict.items():
        tod = meta.get("timeofday", "unknown")
        wth = meta.get("weather", "unknown")
        if tod not in timeofday_subsets:
            timeofday_subsets[tod] = []
        timeofday_subsets[tod].append(img_id)
        if wth not in weather_subsets:
            weather_subsets[wth] = []
        weather_subsets[wth].append(img_id)

    # For the evaluation over subsets, pass the full pred_dict and gt_dict.
    # Note: We add a 'boxes_labels' key to each gt element earlier.
    # Evaluate for time-of-day:
    metrics_tod = {}
    for tod, img_ids in timeofday_subsets.items():
        metrics = evaluate_subset_metrics(
            img_ids, pred_dict, gt_dict, all_class_names, args.iou_threshold
        )
        metrics_tod[tod] = metrics
        print(
            f"TimeOfDay: {tod} -> mAP: {metrics['mAP']:.4f}, AR: {metrics['AR']:.4f}, LocError: {metrics['loc_error']:.4f}, F1: {metrics['F1']:.4f}"
        )

    # Evaluate for weather:
    metrics_weather = {}
    for wth, img_ids in weather_subsets.items():
        metrics = evaluate_subset_metrics(
            img_ids, pred_dict, gt_dict, all_class_names, args.iou_threshold
        )
        metrics_weather[wth] = metrics
        print(
            f"Weather: {wth} -> mAP: {metrics['mAP']:.4f}, AR: {metrics['AR']:.4f}, LocError: {metrics['loc_error']:.4f}, F1: {metrics['F1']:.4f}"
        )

    def plot_bar_metrics(metric_dict, metric_name, title, filename):
        categories = list(metric_dict.keys())
        values = [metric_dict[k][metric_name] for k in categories]
        plt.figure(figsize=(8, 6))
        plt.bar(categories, values, color="skyblue")
        plt.xlabel("Category")
        plt.ylabel(metric_name)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        # plt.show()
        plt.close()

    # Plot time-of-day metrics.
    plot_bar_metrics(metrics_tod, "mAP", "mAP vs Time of Day", "mAP_TimeOfDay.png")
    plot_bar_metrics(metrics_tod, "AR", "AR vs Time of Day", "AR_TimeOfDay.png")
    plot_bar_metrics(
        metrics_tod,
        "loc_error",
        "Localization Error vs Time of Day",
        "LocError_TimeOfDay.png",
    )
    plot_bar_metrics(metrics_tod, "F1", "F1 Score vs Time of Day", "F1_TimeOfDay.png")

    # Plot weather metrics.
    plot_bar_metrics(metrics_weather, "mAP", "mAP vs Weather", "mAP_Weather.png")
    plot_bar_metrics(metrics_weather, "AR", "AR vs Weather", "AR_Weather.png")
    plot_bar_metrics(
        metrics_weather,
        "loc_error",
        "Localization Error vs Weather",
        "LocError_Weather.png",
    )
    plot_bar_metrics(metrics_weather, "F1", "F1 Score vs Weather", "F1_Weather.png")
    print("="*100)
    print("all the plots are saved as png")
    print("="*100)


if __name__ == "__main__":
    main()
