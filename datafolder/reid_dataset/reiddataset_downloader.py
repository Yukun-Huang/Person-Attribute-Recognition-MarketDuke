from __future__ import print_function
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
from .gdrive_downloader import gdrive_downloader
from .cuhk03_to_image import cuhk03_to_image

dataset = {
    'CUHK01': '153IzD3vyQ0PqxxanQRlP9l89F1S5Vr47',
    'CUHK02': '0B2FnquNgAXoneE5YamFXY3NjYWM',
    'CUHK03': '1BO4G9gbOTJgtYIB0VNyHQpZb8Lcn-05m',
    'VIPeR':  '0B2FnquNgAXonZzJPQUtrcWJWbWc',
    'Market1501': '0B2FnquNgAXonU3RTcE1jQlZ3X0E',
    'Market1501Attribute' : '1YMgni5oz-RPkyKHzOKnYRR2H3IRKdsHO',
    'DukeMTMC': '1qtFGJQ6eFu66Tt7WG85KBxtACSE8RBZ0',
    'DukeMTMCAttribute' : '1eilPJFnk_EHECKj2glU_ZLLO7eR3JIiO'
}

dataset_hdf5 = {
    'Market1501': '1ipvyt4qesVK6CUiGcQdwle2c2XYknKco',
    'DukeMTMC': '1tP-fty5YE-W2F6B5rjnQNfE-NzNssGM2'
}

def reiddataset_downloader(data_dir, data_name, hdf5 = True):
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    if hdf5:
        dataset_dir = os.path.join(data_dir , data_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        destination = os.path.join(dataset_dir , data_name+'.hdf5')
        if not os.path.isfile(destination):
            id = dataset_hdf5[data_name]
            print("Downloading %s in HDF5 Formate" %data_name)
            gdrive_downloader(destination, id)
            print("Done")
        else:
            print("Dataset Check Success: %s exists!" %data_name)
    else:
        data_dir_exist = os.path.join(data_dir , data_name)

        if not os.path.exists(data_dir_exist):
            temp_dir = os.path.join(data_dir , 'temp')

            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            destination = os.path.join(temp_dir , data_name)

            id = dataset[data_name]

            print("Downloading %s in Original Images" % data_name)
            gdrive_downloader(destination, id)

            zip_ref = zipfile.ZipFile(destination)
            print("Extracting %s" % data_name)
            zip_ref.extractall(data_dir)
            zip_ref.close()
            shutil.rmtree(temp_dir)
            print("Done")
            if data_name == 'CUHK03':
                print('Converting cuhk03.mat into images')
                cuhk03_to_image(os.path.join(data_dir,'CUHK03'))
                print('Done')
        else:
            print("Dataset Check Success: %s exists!" %data_name)

def reiddataset_downloader_all(data_dir):
    for k,v in dataset.items():
        reiddataset_downloader(k,data_dir)

#For United Testing and External Use
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Name and Dataset Directory')
    parser.add_argument(dest="data_dir", action="store", default="~/Datasets/",help="")
    parser.add_argument(dest="data_name", action="store", type=str,help="")
    args = parser.parse_args() 
    reiddataset_downloader(args.data_dir,args.data_name)