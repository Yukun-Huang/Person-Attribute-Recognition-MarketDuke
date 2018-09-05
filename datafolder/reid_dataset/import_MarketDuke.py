import os
from .reiddataset_downloader import *


def import_MarketDuke(data_dir, dataset_name):
    dataset_dir = os.path.join(data_dir,dataset_name)
    
    if not os.path.exists(dataset_dir):
        print('Please Download '+dataset_name+ ' Dataset')
        
    dataset_dir = os.path.join(data_dir,dataset_name)
    data_group = ['train','query','gallery']
    for group in data_group:
        if group == 'train':
            name_dir = os.path.join(dataset_dir , 'bounding_box_train')
        elif group == 'query':
            name_dir = os.path.join(dataset_dir, 'query')
        else:
            name_dir = os.path.join(dataset_dir, 'bounding_box_test')
        file_list=sorted(os.listdir(name_dir))
        globals()[group]={}
        globals()[group]['data']=[]
        globals()[group]['ids'] = []
        for name in file_list:
            if name[-3:]=='jpg':
                id = name.split('_')[0]
                cam = int(name.split('_')[1][1])
                images = os.path.join(name_dir,name)
                if id not in globals()[group]['ids']:
                    globals()[group]['ids'].append(id)
                globals()[group]['data'].append([images,globals()[group]['ids'].index(id),id,cam,name.split('.')[0]])
    return train,query,gallery