import numpy
import math
import scipy.misc

import numpy
import math
import scipy.misc
import os


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    psnr_rst = 0

    for file in os.listdir('celeba_test'):
        real = scipy.misc.imread(os.path.join('celeba_test', file)).astype(numpy.float32)
        recon = scipy.misc.imread(os.path.join('celeba_eval', file)).astype(numpy.float32)
        psnr_rst += psnr(real,recon)

    print(psnr_rst / len(os.listdir('celeba_test')))
