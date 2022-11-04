import cv2
import numpy as np
import glob
import os
import argparse


def demosaic(raw_image):
    return cv2.cvtColor(raw_image, cv2.COLOR_BAYER_BG2BGR)


def rescale(rgb_image, dtype):
    if dtype == 'float32':
        max = 1.0
    elif dtype == 'uint8':
        max = 255.0
    elif dtype == 'uint16':
        max = 65535.0
    else:
        raise ValueError('not supported data type.')

    # Rescaale pixel value range
    rgb_image = rgb_image.astype(np.float32)
    rgb_image = rgb_image / np.max(rgb_image)
    rgb_image = np.clip(rgb_image, 0.0, 1.0) * max
    rgb_image = rgb_image.astype(dtype)

    return rgb_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, required=True, help='input dir that contains rgb images')
    parser.add_argument('--save_dir', type=str, required=True, help='output dir that saves y, u and v images')
    parser.add_argument('--width', type=int, required=True, help='raw input width')
    parser.add_argument('--height', type=int, required=True, help='raw input height')
    parser.add_argument('--ext', type=str, default='raw', help='search extension')
    parser.add_argument('--data_type', type=str, default='float32', choices=['float32', 'uint8', 'uint16'], help='input data type')
    args = parser.parse_args()

    if args.data_type == 'float32':
        data_type = np.float32
    elif args.data_type == 'uint8':
        data_type = np.uint8
    elif args.data_type == 'uint16':
        data_type = np.uint16
    else:
        raise ValueError('not supported data type.')

    files = sorted(glob.glob(os.path.join(args.load_dir, '*.' + args.ext)))
    for i, file in enumerate(files):
        raw_data = np.fromfile(file, data_type)
        raw_image = raw_data.reshape(args.height, args.width, 1)

        rgb_image = demosaic(raw_image)
        rgb_image = rescale(rgb_image, rgb_image.dtype)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        file_name = os.path.splitext(os.path.basename(file))[0]
        cv2.imwrite(os.path.join(args.save_dir, file_name + '.png'), rgb_image)

        print('{} / {} finished!'.format(i+1, len(files)))
