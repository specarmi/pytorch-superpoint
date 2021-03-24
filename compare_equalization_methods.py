import os
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

base_path_str = '../datasets/FLIR_ADAS/val/Data/'
base_path = Path(base_path_str)
image_paths = list(base_path.iterdir())
image_paths = [str(p) for p in image_paths]

path_str = '../datasets/FLIR_ADAS/val/PreviewData/'
clahe_vals = [40, 300, 1000]
tile_size_vals = [1, 2, 4, 8, 16, 32]
for path in image_paths:
    img_fig, img_axes = plt.subplots(len(tile_size_vals), len(clahe_vals) + 1)
    input_image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    prop_agc = path_str + path[-15:-5] + '.jpeg'
    prop_image = cv2.imread(prop_agc, cv2.IMREAD_ANYDEPTH)
    for j in range(len(tile_size_vals)):
        for i in range(len(clahe_vals)):
            clahe = cv2.createCLAHE(clahe_vals[i], (tile_size_vals[j], tile_size_vals[j]))
            clahe_image = clahe.apply(input_image)
            output_image = cv2.normalize(clahe_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            img_axes[j, i].imshow(output_image, cmap='gray', vmin=0, vmax=255)
            img_axes[j, i].set_title('CLAHE: CL ' + str(clahe_vals[i]) + ', TS ' + str(tile_size_vals[j]) + 'x' + str(tile_size_vals[j]))
            img_axes[j, i].set_xticks([], [])
            img_axes[j, i].set_yticks([], [])
        img_axes[j, -1].imshow(prop_image, cmap='gray', vmin=0, vmax=255)
        img_axes[j, -1].set_title('FLIR PreviewData')
        img_axes[j, -1].set_xticks([], [])
        img_axes[j, -1].set_yticks([], [])
    plt.show()