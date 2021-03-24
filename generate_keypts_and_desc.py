import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from utils.loader import get_module


def read_image(path):
    input_image = cv2.imread(path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    input_image_float = input_image.astype('float32') / 255.0
    H, W = input_image_float.shape[0], input_image_float.shape[1]
    return torch.tensor(input_image_float, dtype=torch.float32).reshape(1, 1, H, W)

def get_pts_desc_from_agent(val_agent, img, device, subpixel, patch_size):
    val_agent.run(img.to(device))

    # heatmap to pts
    pts = val_agent.heatmap_to_pts()
    if subpixel:
        pts = val_agent.soft_argmax_points(pts, patch_size=patch_size)

    # heatmap, pts to desc
    desc_sparse = val_agent.desc_to_sparseDesc()

    return np.asarray(pts[0]).T, np.asarray(desc_sparse[0]).T

if __name__ == "__main__":
    # Handle arguments
    parser = argparse.ArgumentParser(description ='Generate keypoints and descriptors from a folder of images')
    parser.add_argument('config', help = 'the filepath of the config to use', type=str)
    args = parser.parse_args()

    # Import config
    with open(args.config, "r") as f:
        config = yaml.load(f)

    # Set parameters
    subpixel = config["model"]["subpixel"]["enable"]
    patch_size = config["model"]["subpixel"]["patch_size"]

    # Load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Val_model_heatmap = get_module("", config["front_end_model"])
    val_agent = Val_model_heatmap(config["model"], device=device)
    val_agent.loadModel()

    # Create the directory for the keypoints and descriptors
    storage_dir = config['data']['storagepath']
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)

    # Loop over images, generate keypoints and descriptors, and log them
    image_folder = config['data']['filepath']
    for index, image_file in enumerate(sorted(os.listdir(image_folder))):
        image_path = image_folder + image_file

        # Import the image
        img = read_image(image_path)
    
        # Generate keypoints and descriptors
        kpts, desc = get_pts_desc_from_agent(val_agent, img, device, subpixel, patch_size)

        # Keep only the top 300 points (like original bag of binary words paper)
        if config['data']['for_vocab']:
            pts = np.hstack((kpts, desc))
            pts = pts[np.argsort(pts[:, 2])]
            kpts = pts[-300:, :3]
            desc = pts[-300:, 3:]
    
        # Write the results to a yaml file
        result_file = cv2.FileStorage(storage_dir + str(index + 1) + '.yaml', 1)
        result_file.write('keypoints', kpts)
        result_file.write('descriptors', desc)
        result_file.release()