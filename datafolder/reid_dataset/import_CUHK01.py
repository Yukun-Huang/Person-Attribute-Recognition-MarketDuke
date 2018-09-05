import os
from .reiddataset_downloader import *
def import_CUHK01(dataset_dir):
    cuhk01_dir = os.path.join(dataset_dir,'CUHK01')
    
    if not os.path.exists(cuhk01_dir):
        print('Please Download the CUHK01 Dataset')
    
    file_list=os.listdir(cuhk01_dir)
    name_dict={}
    for name in file_list:
        if name[-3:]=='png':
            id = name[:4]
            if id not in name_dict:
                name_dict[id]=[]
                name_dict[id].append([])
                name_dict[id].append([])
            if int(name[-7:-4])<3:
                name_dict[id][0].append(os.path.join(cuhk01_dir,name))  
            else:
                name_dict[id][1].append(os.path.join(cuhk01_dir,name))  
    return name_dict