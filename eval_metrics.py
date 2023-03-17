import argparse
import cv2
from math import log10, sqrt
import math
import numpy as np

# Assuming image extension was 1/4 of image width

parser = argparse.ArgumentParser()
parser.add_argument('--changed_image', default='', type=str,
                    help='The Changed Image Name')
parser.add_argument('--gt', default='', type=str,
                    help='Image without the change (ground truth).')
parser.add_argument('--output', default='./metrics.txt', type=str,
                    help='Output file.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    #  A value closer to 1 indicates better image quality.


def PSNR(original, changed):
    mse = np.mean((original - changed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
# Good value is between 30 and 50.



if __name__ == "__main__":
    args = parser.parse_args()

    image = cv2.imread(args.changed_image)
    original = cv2.imread(args.gt)

    width = original.shape[1]

    image = image[:, width:]
    flipped = cv2.flip(original, 1)
    flippedCropped = flipped[:, :int(width/4)]



    fo = open(args.output, "w")
    fo.write("PSNR: {}\n".format(PSNR(flippedCropped, image)))
    fo.write("SSIM: {}\n".format(calculate_ssim(flippedCropped, image)))
    fo.close()
