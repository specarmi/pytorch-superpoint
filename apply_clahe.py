import os
import sys
from pathlib import Path
import cv2

base_path_str = 'FLIR_ADAS/val/Data/'
base_path = Path(base_path_str)
image_paths = list(base_path.iterdir())
image_paths = [str(p) for p in image_paths]

# Create a new directory
results_dir = base_path_str[:-1] + "_CLAHE_AGC"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

clahe = cv2.createCLAHE(5, (8, 8))
for path in image_paths:
    input_image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    input_image = cv2.normalize(input_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    output_image = clahe.apply(input_image)
    cv2.imwrite(results_dir + '/' + path[-10:-5] + '.png', output_image)
