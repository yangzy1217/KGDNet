import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
from tqdm import tqdm

import torch

from depth_anything_v2.dpt import DepthAnythingV2 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Prediction for All Images in a Directory')

    parser.add_argument('--img-dir', type=str, required=True, help='Path to the directory containing images')
    parser.add_argument('--input-size', type=int, default=1024, help='Input size for the model')
    parser.add_argument('--outdir', type=str, required=True, help='Directory to save depth images')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'], help='Encoder type')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the Depth Anything V2 checkpoint')
    parser.add_argument('--grayscale', dest='grayscale', default=False, action='store_true', help='Save depth images as grayscale')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    if not os.path.isdir(args.img_dir):
        raise ValueError(f"The specified path {args.img_dir} is not a directory.")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    filenames = glob.glob(os.path.join(args.img_dir, '**/*'), recursive=True)
    filenames = [f for f in filenames if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif'))]

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for filename in tqdm(filenames, desc="Processing images"):
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"Warning: Unable to read image {filename}, skipping.")
            continue

        depth = depth_anything.infer_image(raw_image, args.input_size)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        base_filename = os.path.splitext(os.path.basename(filename))[0]
        output_filename = os.path.join(args.outdir, f'{base_filename}.png')

        cv2.imwrite(output_filename, depth)

    print(f"Processing complete. Depth images saved to {args.outdir}.")

# /raid/YZY/Dataset/SYSU-CD/train/A
# /raid/YZY/Dataset/ZJCD_split/train/A
