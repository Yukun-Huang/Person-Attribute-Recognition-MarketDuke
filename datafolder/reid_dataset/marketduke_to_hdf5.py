import warnings
warnings.filterwarnings('ignore','.*conversion.*')

import os
import h5py
import numpy as np
from PIL import  Image
from .import_MarketDuke import import_MarketDuke

def marketduke_to_hdf5(data_dir,dataset_name,save_dir=os.getcwd()):
    phase_list = ['train','query','gallery']
    dataset = import_MarketDuke(data_dir,dataset_name)
    dt = h5py.special_dtype(vlen=str)
    
    f = h5py.File(os.path.join(save_dir,dataset_name+'.hdf5'),'w')
    for phase in phase_list:
        grp = f.create_group(phase)
        phase_dataset = dataset[phase_list.index(phase)]
        for i in range(len(phase_dataset['data'])):
            name = phase_dataset['data'][i][0].split('/')[-1].split('.')[0]
            temp = grp.create_group(name) 
            temp.create_dataset('img',data=Image.open(phase_dataset['data'][i][0]))
            temp.create_dataset('index',data=int(phase_dataset['data'][i][1]))
            temp.create_dataset('id',data=phase_dataset['data'][i][2], dtype=dt)
            temp.create_dataset('cam',data=int(phase_dataset['data'][i][3]))
        
    ids = f.create_group('ids')
    ids.create_dataset('train',data=np.array(dataset[0]['ids'],'S4'),dtype=dt)
    ids.create_dataset('query',data=np.array(dataset[1]['ids'],'S4'),dtype=dt)
    ids.create_dataset('gallery',data=np.array(dataset[2]['ids'],'S4'),dtype=dt)
    
    f.close()