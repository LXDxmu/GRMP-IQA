# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:18:18 2019

@author: Administrator
"""

import numpy as np
from scipy import io as sio
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import cv2
import os
import csv
import pandas as pd

name = sio.loadmat('/data/IQA-dataset/ChallengeDB_release/Data/AllImages_release.mat')['AllImages_release']
num = len(name)
mos = sio.loadmat('/data/IQA-dataset/ChallengeDB_release/Data/AllMOS_release.mat')["AllMOS_release"][0].astype(np.float32)

ind = np.arange(num)
ind_train = ind[:int(len(ind) * 0.8)]
ind_test = ind[int(len(ind) * 0.8):]

imgs_all = np.zeros((num, 3, 244, 244), dtype=np.uint8)


for i in np.arange(0, num):
    impath = "/data/IQA-dataset/ChallengeDB_release/Images"
    if i<7:
        impath = impath + "/" + "trainingImages" + "/" + str(name[i][0][0])
    else:
        impath = impath + "/" + str(name[i][0][0])

    im = cv2.cvtColor(cv2.imread(impath, 1), cv2.COLOR_BGR2RGB)
    imgs_all[i] = cv2.resize(im, (244, 244), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

sio.savemat('livew_244.mat', {'X': imgs_all[ind_train], 'Y': mos[ind_train], 'Xtest': imgs_all[ind_test], 'Ytest': mos[ind_test]})

