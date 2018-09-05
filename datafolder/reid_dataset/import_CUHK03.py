import os
from .reiddataset_downloader import *
def import_CUHK03(dataset_dir, detected = False):
    
    cuhk03_dir = os.path.join(dataset_dir,'CUHK03')
    
    if not os.path.exists(cuhk03_dir):
        Print('Please Download the CUHK03 Dataset')
    
    if not detected:
        cuhk03_dir = os.path.join(cuhk03_dir , 'labeled')
    else:
        cuhk03_dir = os.path.join(cuhk03_dir , 'detected')

    campair_list = os.listdir(cuhk03_dir)
    #campair_list = ['P1','P2','P3']
    name_dict={}
    for campair in campair_list:
        cam1_list = []
        cam1_list=os.listdir(os.path.join(cuhk03_dir,campair,'cam1'))
        cam2_list=os.listdir(os.path.join(cuhk03_dir,campair,'cam2'))
        for file in cam1_list:
            id = campair[1:]+'-'+file.split('-')[0]
            if id not in name_dict:
                name_dict[id]=[]
                name_dict[id].append([])
                name_dict[id].append([])
            name_dict[id][0].append(os.path.join(cuhk03_dir,campair,'cam1',file))
        for file in cam2_list:
            id = campair[1:]+'-'+file.split('-')[0]
            if id not in name_dict:
                name_dict[id]=[]
                name_dict[id].append([])
                name_dict[id].append([])
            name_dict[id][1].append(os.path.join(cuhk03_dir,campair,'cam2',file))
    return name_dict

def cuhk03_test(data_dir):
    CUHK03_dir = os.path.join(data_dir , 'CUHK03')
    f = h5py.File(os.path.join(CUHK03_dir,'cuhk-03.mat'))
    test = []
    for i in range(20):
        test_set = (np.array(f[f['testsets'][0][i]],dtype='int').T).tolist()
        test.append(test_set)

        return test