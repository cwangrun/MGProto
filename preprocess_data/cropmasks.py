import os
import pandas as pd
from PIL import Image
from shutil import copyfile
import numpy as np


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# set paths
rootpath = '/mnt/c/chong/data/CUB_200_2011/CUB_200_2011/'
# imgspath = rootpath + 'images/'
imgspath = '/home/chong/Downloads/segmentations/'
trainpath = '/home/chong/Downloads/mask_train/'
testpath = '/home/chong/Downloads/mask_test/'

# read img names, bounding_boxes
names = pd.read_table(rootpath + 'images.txt', delimiter=' ', names=['id', 'name'])
names = names.to_numpy()
boxs = pd.read_table(rootpath + 'bounding_boxes.txt', delimiter=' ',names=['id', 'x', 'y', 'width', 'height'])
boxs = boxs.to_numpy()


# crop imgs
for i in range(11788):
    mask_path = (imgspath + names[i][1]).replace('.jpg', '.png')
    im = Image.open(mask_path)
    im = im.crop((boxs[i][1], boxs[i][2], boxs[i][1] + boxs[i][3], boxs[i][2] + boxs[i][4]))
    im.save(mask_path, quality=100)
    print('{} masks cropped and saved.'.format(i + 1))
print('All Done.')

# mkdir for cropped masks
folders = pd.read_table(rootpath + 'classes.txt', delimiter=' ', names=['id', 'folder'])
folders = folders.to_numpy()
for i in range(200):
    makedir(trainpath + folders[i][1]) #200
    makedir(testpath + folders[i][1])

# split imgs
labels = pd.read_table(rootpath + 'train_test_split.txt', delimiter=' ', names=['id', 'label'])
labels = labels.to_numpy()
for i in range(11788):
    mask_path = (imgspath + names[i][1]).replace('.jpg', '.png')
    save_name = (names[i][1]).replace('.jpg', '.png')
    if(labels[i][1] == 1):
        copyfile(mask_path, trainpath + save_name)
    else:
        copyfile(mask_path, testpath + save_name)
    print('{} mask splited.'.format(i + 1))
print('All Done.')
