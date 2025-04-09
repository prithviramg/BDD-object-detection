import os
import json
import math
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Define your class mappings
tc_classes = {"traffic light": 0, "traffic sign": 1}
tp_classes = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motor": 6,
    "bike": 7,
}


def gaussian2D(shape, sigma=1):
    """Generate a 2D gaussian kernel."""
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    """Draw a 2D gaussian on the heatmap."""
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[:2]

    left = min(x, radius)
    right = min(width - x, radius + 1)
    top = min(y, radius)
    bottom = min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def gaussian_radius(det_size, min_overlap=0.7):
    """Compute gaussian radius for an object of size det_size."""
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return int(min(r1, r2, r3))


class BDDDataset(Dataset):
    def __init__(self, image_dir, annotation_file, input_size=(512, 512), max_objs=128):
        """
        image_dir: Directory where images are stored.
        annotation_file: JSON file containing the BDD annotations.
        input_size: Desired input image size as (height, width).
        max_objs: Maximum number of objects per image.
        """
        self.image_dir = image_dir
        self.input_size = input_size  # (H, W)
        self.max_objs = max_objs

        # For TC (traffic control objects: traffic light and sign)
        self.tc_classes = tc_classes
        self.num_tc = len(tc_classes)
        self.down_ratio_tc = 4  # Use P3: feature map with resolution input_size/2
        self.output_size_tc = (
            self.input_size[0] // self.down_ratio_tc,
            self.input_size[1] // self.down_ratio_tc,
        )

        # For TP (traffic participant objects)
        self.tp_classes = tp_classes
        self.num_tp = len(tp_classes)
        self.down_ratio_tp = 8  # Use P4: feature map with resolution input_size/4
        self.output_size_tp = (
            self.input_size[0] // self.down_ratio_tp,
            self.input_size[1] // self.down_ratio_tp,
        )

        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

        # Basic transforms: resize image and convert to tensor.
        self.img_transform = transforms.Compose(
            [
                transforms.Resize(self.input_size, interpolation=Image.BILINEAR),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Each annotation entry should have 'name' and 'labels'
        ann = self.annotations[index]
        img_path = os.path.join(self.image_dir, ann["name"])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize image
        img = self.img_transform(img)

        # Calculate scaling factors for the resized image (input_size)
        scale_x = self.input_size[1] / orig_w
        scale_y = self.input_size[0] / orig_h

        # Prepare target arrays for TC branch (P3: input_size/4), since we have small objects
        H_tc, W_tc = self.output_size_tc
        tc_heatmap = np.zeros((self.num_tc, H_tc, W_tc), dtype=np.float32)
        tc_wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        tc_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        tc_reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        tc_ind = np.zeros((self.max_objs), dtype=np.int64)
        tc_count = 0

        # Prepare target arrays for TP branch (P4: input_size/8), since we have large objects
        H_tp, W_tp = self.output_size_tp
        tp_heatmap = np.zeros((self.num_tp, H_tp, W_tp), dtype=np.float32)
        tp_wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        tp_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        tp_reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        tp_ind = np.zeros((self.max_objs), dtype=np.int64)
        tp_count = 0

        for obj in ann["labels"]:
            category = obj["category"]
            if "box2d" not in obj:
                continue
            box = obj["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y
            w_box = x2 - x1
            h_box = y2 - y1
            if w_box <= 0 or h_box <= 0:
                continue

            if category in self.tc_classes:
                factor = self.down_ratio_tc
                out_W, out_H = W_tc, H_tc

                # Compute center in feature map coordinates
                ct_x = (x1 + w_box / 2) / factor
                ct_y = (y1 + h_box / 2) / factor
                ct = np.array([ct_x, ct_y], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                # Compute width and height in the feature map space
                w_feat = (w_box / factor) / W_tc
                h_feat = (h_box / factor) / H_tc
                radius = gaussian_radius((math.ceil(h_feat), math.ceil(w_feat)))
                radius = max(0, int(radius))

                cls_id = self.tc_classes[category]
                draw_gaussian(tc_heatmap[cls_id], ct_int, radius)
                if tc_count < self.max_objs:
                    tc_wh[tc_count] = np.array([w_feat, h_feat], dtype=np.float32)
                    tc_reg[tc_count] = ct - ct_int
                    tc_reg_mask[tc_count] = 1
                    tc_ind[tc_count] = ct_int[1] * out_W + ct_int[0]
                    tc_count += 1

            # Process traffic participant objects (TP) using down_ratio_tp
            elif category in self.tp_classes:
                factor = self.down_ratio_tp
                out_W, out_H = W_tp, H_tp

                ct_x = (x1 + w_box / 2) / factor
                ct_y = (y1 + h_box / 2) / factor
                ct = np.array([ct_x, ct_y], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                w_feat = (w_box / factor) / W_tp
                h_feat = (h_box / factor) / H_tp
                radius = gaussian_radius((math.ceil(h_feat), math.ceil(w_feat)))
                radius = max(0, int(radius))

                cls_id = self.tp_classes[category]
                draw_gaussian(tp_heatmap[cls_id], ct_int, radius)
                if tp_count < self.max_objs:
                    tp_wh[tp_count] = np.array([w_feat, h_feat], dtype=np.float32)
                    tp_reg[tp_count] = ct - ct_int
                    tp_reg_mask[tp_count] = 1
                    tp_ind[tp_count] = ct_int[1] * out_W + ct_int[0]
                    tp_count += 1

        # Convert target arrays to tensors.
        targets = {
            "tc": {
                "heatmap": torch.tensor(tc_heatmap),
                "wh": torch.tensor(tc_wh),
                "reg": torch.tensor(tc_reg),
                "reg_mask": torch.tensor(tc_reg_mask, dtype=torch.uint8),
                "ind": torch.tensor(tc_ind, dtype=torch.long),
            },
            "tp": {
                "heatmap": torch.tensor(tp_heatmap),
                "wh": torch.tensor(tp_wh),
                "reg": torch.tensor(tp_reg),
                "reg_mask": torch.tensor(tp_reg_mask, dtype=torch.uint8),
                "ind": torch.tensor(tp_ind, dtype=torch.long),
            },
        }
        return {"image": img, "targets": targets}


def collate_fn(batch):
    """
    Custom collate function for the BDDDataset.
    Each sample is a dict containing:
      - "image": tensor of shape [3, H, W]
      - "targets": a dict with two keys:
           "tc": { "heatmap": [C, H1, W1], "wh": [max_objs, 2], "reg": [max_objs, 2],
                   "reg_mask": [max_objs], "ind": [max_objs] }
           "tp": { "heatmap": [C, H2, W2], "wh": [max_objs, 2], "reg": [max_objs, 2],
                   "reg_mask": [max_objs], "ind": [max_objs] }
    This function stacks images and each target component in the batch.
    """
    images = [sample["image"] for sample in batch]
    images = torch.stack(images, dim=0)

    # For targets, we need to combine each sub-key from "tc" and "tp" separately.
    def stack_targets(key):
        return torch.stack([sample["targets"][key] for sample in batch], dim=0)

    targets = {}
    # For traffic control (TC) branch
    targets["tc"] = {}
    for sub_key in batch[0]["targets"]["tc"]:
        targets["tc"][sub_key] = (
            stack_targets(lambda sample=None, key=sub_key: sample["targets"]["tc"][key])
            if False
            else torch.stack(
                [sample["targets"]["tc"][sub_key] for sample in batch], dim=0
            )
        )
    # For traffic participant (TP) branch
    targets["tp"] = {}
    for sub_key in batch[0]["targets"]["tp"]:
        targets["tp"][sub_key] = torch.stack(
            [sample["targets"]["tp"][sub_key] for sample in batch], dim=0
        )

    return images, targets


if __name__ == "__main__":
    # Set your dataset paths accordingly.
    image_directory = (
        r"D:\database\bdd_dataset\bdd100k_images_100k\bdd100k\images\100k\train"
    )
    annotation_file = r"D:\database\bdd_dataset\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json"

    # Create an instance of the BDDDataset.
    dataset = BDDDataset(
        image_dir=image_directory,
        annotation_file=annotation_file,
        input_size=(720, 1280),
        max_objs=64,
    )

    # Create a DataLoader with the custom collate function.
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Test one batch
    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        print("  Images shape:", images.shape)
        print("  TC Heatmap shape:", targets["tc"]["heatmap"].shape)
        print("  TP Heatmap shape:", targets["tp"]["heatmap"].shape)
        # Print additional info if needed:
        print("  TC WH shape:", targets["tc"]["wh"].shape)
        print("  TP WH shape:", targets["tp"]["wh"].shape)
        break
