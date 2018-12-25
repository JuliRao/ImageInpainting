from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from model import _netG

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--test_image', default='mnist_test', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='model/netG_streetview.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=32,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.999,help='0 means do not use else use with this weight')
opt = parser.parse_args()
print(opt)


def load_pic(path, grayscale=False):
    pic = utils.load_image(path, opt.imageSize)
    if grayscale:
        transform = transforms.Compose([
            transforms.Scale(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Scale(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
        ])
    pic = transform(pic)
    return pic


origin_mask = load_pic('designed_mask.png', True).byte()


try:
    os.mkdir('celeba_eval')
except OSError:
    pass


dir = os.listdir('celeba_test')
num_pics = len(dir)
whole_pic = torch.FloatTensor(num_pics, opt.nc, opt.imageSize, opt.imageSize)


for i in range(num_pics):
    file = str(i+99997) + '.jpg'
    image = load_pic(os.path.join('celeba_test', file))
    image = image.unsqueeze(0)
    mask = origin_mask.expand_as(image)

    input_cropped = torch.FloatTensor(1, 3, opt.imageSize, opt.imageSize)
    input_cropped.data.resize_(image.size()).copy_(image)
    input_cropped[mask] = 1

    vutils.save_image(input_cropped[0], os.path.join('celeba_eval', 'c_'+file))
