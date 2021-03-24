import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def read_image(path):
    input_image = cv2.imread(path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    input_image_float = input_image.astype('float32') / 255.0
    H, W = input_image_float.shape[0], input_image_float.shape[1]
    return input_image, torch.tensor(input_image_float, dtype=torch.float32).reshape(1, 1, H, W)

def get_pts_desc_from_agent(val_agent, img, device, subpixel, patch_size):
    val_agent.run(img.to(device))

    # heatmap to pts
    pts = val_agent.heatmap_to_pts()
    if subpixel:
        pts = val_agent.soft_argmax_points(pts, patch_size=patch_size)

    # heatmap, pts to desc
    desc_sparse = val_agent.desc_to_sparseDesc()

    return np.asarray(pts[0]), np.asarray(desc_sparse[0])


def plot_superpoint_matches(config, img_1, img_2, img_1_tensor, img_2_tensor):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # parameters
    subpixel = config["model"]["subpixel"]["enable"]
    patch_size = config["model"]["subpixel"]["patch_size"]

    # model loading
    from utils.loader import get_module
    Val_model_heatmap = get_module("", config["front_end_model"])
    val_agent = Val_model_heatmap(config["model"], device=device)
    val_agent.loadModel()

    # first image
    pts_1, desc_1 = get_pts_desc_from_agent(val_agent, img_1_tensor, device, subpixel, patch_size)
    kpts_1 = [cv2.KeyPoint(pt[0], pt[1], pt[2]) for pt in pts_1.T]

    # second image
    pts_2, desc_2 = get_pts_desc_from_agent(val_agent, img_2_tensor, device, subpixel, patch_size)
    kpts_2 = [cv2.KeyPoint(pt[0], pt[1], pt[2]) for pt in pts_2.T]

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(desc_1.T, desc_2.T)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw all matches
    print(len(matches))
    img_3 = cv2.drawMatches(img_1, kpts_1, img_2, kpts_2, matches[:100], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_3)
    plt.show()

def plot_orb_matches(img_1, img_2):
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kpts_1, desc_1 = orb.detectAndCompute(img_1, None)
    kpts_2, desc_2 = orb.detectAndCompute(img_2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(desc_1, desc_2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Draw all matches
    print(len(matches))
    img_3 = cv2.drawMatches(img_1, kpts_1, img_2, kpts_2, matches[:100], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_3)
    plt.show()


if __name__ == "__main__":
    # Handle arguments
    parser = argparse.ArgumentParser(description ='Draw matches between two images using various features')
    parser.add_argument('config', help = 'the filepath of the config to use for superpoint', type=str)
    parser.add_argument('img_1', help = 'filename of the first image', type=str)
    parser.add_argument('img_2', help = 'filename of the second image', type=str)
    args = parser.parse_args()

    # Import config
    with open(args.config, "r") as f:
        config = yaml.load(f)
    print("Config:", config)

    # Import images
    img_1_path = config['data']['filepath'] + args.img_1
    img_2_path = config['data']['filepath'] + args.img_2
    img_1, img_1_tensor = read_image(img_1_path) 
    img_2, img_2_tensor = read_image(img_2_path)

    # Plot matches
    plot_superpoint_matches(config, img_1, img_2, img_1_tensor, img_2_tensor)
    plot_orb_matches(img_1, img_2)