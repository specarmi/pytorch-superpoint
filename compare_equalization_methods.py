import os
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

base_path_str = 'FLIR_ADAS/val/Data/'
base_path = Path(base_path_str)
image_paths = list(base_path.iterdir())
image_paths = [str(p) for p in image_paths]

path_str = 'FLIR_ADAS/val/PreviewData/'
clahe_vals = [1, 40, 100, 200, 300, 500]
for path in image_paths:
    img_fig, img_axes = plt.subplots(1, len(clahe_vals) + 1)
    input_image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    for i in range(len(clahe_vals)):
        clahe = cv2.createCLAHE(clahe_vals[i], (8, 8))
        clahe_image = clahe.apply(input_image)
        output_image = cv2.normalize(clahe_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        img_axes[i].imshow(output_image, cmap='gray', vmin=0, vmax=255)
        img_axes[i].set_title('CLAHE: Clip Limit ' + str(clahe_vals[i]))
    prop_agc = path_str + path[-15:-5] + '.jpeg'
    print(prop_agc)
    prop_image = cv2.imread(prop_agc, cv2.IMREAD_ANYDEPTH)
    img_axes[-1].imshow(prop_image, cmap='gray', vmin=0, vmax=255)
    img_axes[-1].set_title('FLIR PreviewData')
    plt.show()