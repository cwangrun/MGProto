import os
import pandas as pd
from PIL import Image
from shutil import copyfile


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


img_dir = '/home/chong/Downloads/oxford-iiit-pet/images'

label_file = '/home/chong/Downloads/oxford-iiit-pet/annotations/trainval.txt'
savepath = '/mnt/c/chong/data/Pets_full/train/'


# read img lists
img_list = open(label_file, "r").readlines()

# crop imgs
for line in img_list:
    info = line.strip().split(' ')
    src_path = os.path.join(img_dir, info[0] + '.jpg')
    dst_path = os.path.join(savepath, info[1], info[0] + '.jpg')
    makedir(os.path.dirname(dst_path))
    copyfile(src_path, dst_path)
print('All Done.')




