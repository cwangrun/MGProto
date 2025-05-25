import os
import pandas as pd
from PIL import Image
from shutil import copyfile
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


path_main = '/mnt/c/chong/data/CUB_200_2011_full/mask_test'
save_main = '/mnt/c/chong/data/CUB_200_2011_full/mask_test_fg'

mask_path_all = glob(path_main + '/*/*.png')

# crop imgs
for mask_path in mask_path_all:
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)

    label = np.sort(np.unique(mask))

    print(label, )

    mask_new = np.logical_or(mask == label[0], mask == label[1])  # 0 and 51
    mask_new = np.logical_not(mask_new)

    mask_new = np.uint8(mask_new * 255)

    save_path = mask_path.replace(path_main, save_main)
    makedir(os.path.dirname(save_path))

    mask_new = Image.fromarray(mask_new)
    mask_new.save(save_path, quality=100)

print('All Done.')


