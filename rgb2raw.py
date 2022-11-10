import cv2
import numpy as np
import argparse
import glob
import os
from PIL import Image


def flatten_raw_image(raw_image_4ch):
    out = np.zeros_like(
        raw_image_4ch, 
        shape=(raw_image_4ch.shape[1] * 2, raw_image_4ch.shape[2] * 2)
    )

    out[0::2, 0::2] = raw_image_4ch[0, :, :]
    out[0::2, 1::2] = raw_image_4ch[1, :, :]
    out[1::2, 0::2] = raw_image_4ch[2, :, :]
    out[1::2, 1::2] = raw_image_4ch[3, :, :]

    return out


def mosaic(rgb_image):
    red = rgb_image[0::2, 0::2, 0]
    green_red = rgb_image[0::2, 1::2, 1]
    green_blue = rgb_image[1::2, 0::2, 1]
    blue = rgb_image[1::2, 1::2, 2]
    rgb_stack = np.stack([red, green_red, green_blue, blue])
    
    return flatten_raw_image(rgb_stack)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, required=True, help='input dir that contains rgb images')
    parser.add_argument('--save_dir', type=str, required=True, help='output dir that saves y, u and v images')
    parser.add_argument('--ext', type=str, default='png', help='search extension')
    parser.add_argument('--data_type', type=str, default='float32', choices=['float32', 'uint8', 'uint16'], help='output data type')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.load_dir, '*.' + args.ext)))
    for i, file in enumerate(files):
        rgb_image = Image.open(file).convert('RGB')
        rgb_image = np.array(rgb_image)

        raw_image = mosaic(rgb_image)
        if args.data_type == 'float32':
            raw_image = raw_image.astype(np.float32) / 255.0
        elif args.data_type == 'uint8':
            raw_image = raw_image.astype(np.uint8)
        elif args.data_type == 'uint16':
            raw_image = raw_image.astype(np.float32) / 255.0 * 65535.0
            raw_image = raw_image.astype(np.uint16)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        file_name = os.path.splitext(os.path.basename(file))[0]
        raw_image.tofile(os.path.join(args.save_dir, file_name + '.raw'))

        print('{} / {} finished!'.format(i+1, len(files)))
