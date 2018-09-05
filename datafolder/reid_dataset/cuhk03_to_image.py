import warnings
warnings.filterwarnings('ignore','.*conversion.*')

import os
import zipfile
import shutil
import requests
import h5py
import numpy as np
from PIL import Image
import argparse

def cuhk03_to_image(CUHK03_dir):
    
    f = h5py.File(os.path.join(CUHK03_dir,'cuhk-03.mat'))

    detected_labeled = ['detected','labeled']
    print('converting')
    for data_type in detected_labeled:

        datatype_dir = os.path.join(CUHK03_dir, data_type)
        if not os.path.exists(datatype_dir):
                os.makedirs(datatype_dir)

        for campair in range(len(f[data_type][0])):
            campair_dir = os.path.join(datatype_dir,'P%d'%(campair+1))
            cam1_dir = os.path.join(campair_dir,'cam1')
            cam2_dir = os.path.join(campair_dir,'cam2')

            if not os.path.exists(campair_dir):
                os.makedirs(campair_dir)
            if not os.path.exists(cam1_dir):
                os.makedirs(cam1_dir)
            if not os.path.exists(cam2_dir):
                os.makedirs(cam2_dir)

            for img_no in range(f[f[data_type][0][campair]].shape[0]):
                if img_no < 5:
                    cam_dir = 'cam1'
                else:
                    cam_dir = 'cam2'
                for person_id in range(f[f[data_type][0][campair]].shape[1]):
                    img = np.array(f[f[f[data_type][0][campair]][img_no][person_id]])
                    if img.shape[0] !=2:
                        img = np.transpose(img, (2,1,0))
                        im = Image.fromarray(img)
                        im.save(os.path.join(campair_dir, cam_dir, "%d-%d.jpg"%(person_id+1,img_no+1)))