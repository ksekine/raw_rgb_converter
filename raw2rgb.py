import cv2
import numpy as np
import glob
import os
import argparse


def pack_raw_image(raw_image):
    out = np.zeros_like(
        raw_image, 
        shape=(4, raw_image.shape[0] // 2, raw_image.shape[1] // 2)
    )

    out[0, :, :] = raw_image[0::2, 0::2]
    out[1, :, :] = raw_image[0::2, 1::2]
    out[2, :, :] = raw_image[1::2, 0::2]
    out[3, :, :] = raw_image[1::2, 1::2]

    return out


def demosaic(image, mode='linear'):
    # assert isinstance(image, torch.Tensor)
    image = image.clamp(0.0, 1.0) * 255

    if image.dim() == 4:
        num_images = image.dim()
        batch_input = True
    else:
        num_images = 1
        batch_input = False
        image = image.unsqueeze(0)  # [1, 4, 32, 32]

    # padding （Bayer2RGBの出力がエッジ付近でおかしくなるため）
    image = F.pad(image, (4, 4, 4, 4), "reflect")

    # Generate single channel input for opencv
    im_sc = torch.zeros((num_images, image.shape[-2] * 2, image.shape[-1] * 2, 1))
    im_sc[:, ::2, ::2, 0] = image[:, 0, :, :]
    im_sc[:, ::2, 1::2, 0] = image[:, 1, :, :]
    im_sc[:, 1::2, ::2, 0] = image[:, 2, :, :]
    im_sc[:, 1::2, 1::2, 0] = image[:, 3, :, :]

    im_sc = im_sc.numpy().astype(np.uint8)

    out = []

    def shift(image, shifts=[0, 0]):  # add
        h, w = image.shape[:2]
        M = np.float32([[1, 0, shifts[0]], [0, 1, shifts[1]]])
        return cv2.warpAffine(image, M, (w, h))

    for im in im_sc:
        # cv.imwrite('frames/tmp.png', im)
        if mode == 'linear':
            im_dem_np = cv2.cvtColor(im, cv2.COLOR_BAYER_BG2RGB)
            im_dem_np = shift(im_dem_np, [0, 0])  # add
        elif mode == 'vng':
            im_dem_np = cv2.cvtColor(im, cv2.COLOR_BAYER_BG2RGB_VNG)
            im_dem_np = shift(im_dem_np, [0, 2])  # add
        elif mode == 'ea':
            im_dem_np = cv2.cvtColor(im, cv2.COLOR_BAYER_BG2RGB_EA)
            im_dem_np = shift(im_dem_np, [0, 0])  # add
        im_dem_np_ = im_dem_np[8:-8, 8:-8, :]  # unpadding


def demosaic2(raw_image):
    raw_image = pack_raw_image(raw_image)
    raw_image = np.clip(raw_image, 0.0, 1.0) * 255.0

    raw_image_sc = np.zeros((raw_image.shape[-2] * 2, raw_image.shape[-1] * 2, 1))
    raw_image_sc[::2, ::2, 0] = raw_image[0, :, :]
    raw_image_sc[::2, 1::2, 0] = raw_image[1, :, :]
    raw_image_sc[1::2, ::2, 0] = raw_image[2, :, :]
    raw_image_sc[1::2, 1::2, 0] = raw_image[3, :, :]
    raw_image_sc = raw_image_sc.astype(np.uint8)

    rgb_image = cv2.cvtColor(raw_image_sc, cv2.COLOR_BAYER_BG2BGR)
    rgb_image = rgb_image.astype(np.float32)
    rgb_image = rgb_image / np.max(rgb_image)
    rgb_image = np.clip(rgb_image, 0.0, 1.0) * 255.0
    rgb_image = rgb_image.astype(np.uint8)

    return rgb_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, required=True, help='input dir that contains rgb images')
    parser.add_argument('--save_dir', type=str, required=True, help='output dir that saves y, u and v images')
    parser.add_argument('--width', type=int, required=True, help='raw input width')
    parser.add_argument('--height', type=int, required=True, help='raw input height')
    parser.add_argument('--ext', type=str, default='raw', help='search extension')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.load_dir, '*.' + args.ext)))
    for i, file in enumerate(files):
        raw_data = np.fromfile(file, np.float32)
        raw_image = raw_data.reshape(args.width, args.height)
        # out = demosaic(pack_raw_image(data))
        rgb_image = demosaic2(raw_image)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        file_name = os.path.splitext(os.path.basename(file))[0]
        cv2.imwrite(os.path.join(args.save_dir, file_name + '.png'), rgb_image)

        print('{} / {} finished!'.format(i+1, len(files)))
