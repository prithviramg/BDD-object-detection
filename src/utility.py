import numpy as np
import cv2


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    return max(0, int((b1 + sq1) / 2))


def draw_gaussian(heatmap, center, radius):
    diameter = 2 * radius + 1
    gaussian = cv2.getGaussianKernel(diameter, sigma=diameter / 6)
    gaussian = gaussian @ gaussian.T

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]

    if masked_heatmap.shape == masked_gaussian.shape:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
