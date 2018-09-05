import os
from .reiddataset_downloader import *
def import_VIPeR(dataset_dir):
    viper_dir = os.path.join(dataset_dir , 'VIPeR')
    if not os.path.exists(viper_dir):
        orint('Please Download VIPeR Dataset')
        
    file_list_a=os.listdir(os.path.join(viper_dir,'cam_a'))
    file_list_b=os.listdir(os.path.join(viper_dir,'cam_b'))
    
    name_dict={}
    for name in file_list_a:
        if name[-3:]=='bmp':
            id = name.split('_')[0]
            if id not in name_dict:
                name_dict[id]=[]
                name_dict[id].append([])
                name_dict[id].append([])
            name_dict[id][0].append(os.path.join(viper_dir,'cam_a',name))
    for name in file_list_b:
        if name[-3:]=='bmp':
            id = name.split('_')[0]
            if id not in name_dict:
                name_dict[id]=[]
                name_dict[id].append([])
                name_dict[id].append([])
            name_dict[id][1].append(os.path.join(viper_dir,'cam_b',name))
                
    return name_dict