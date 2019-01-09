import os
from .import_Market1501 import *
from .reiddataset_downloader import *
import scipy.io


def import_Market1501Attribute(dataset_dir):
    dataset_name = 'Market-1501/attribute'
    train,query,test = import_Market1501(dataset_dir)
    if not os.path.exists(os.path.join(dataset_dir,dataset_name)):
        print('Please Download the Market1501Attribute Dataset')
    train_label=['age',
           'backpack',
           'bag',
           'handbag',
           'downblack',
           'downblue',
           'downbrown',
           'downgray',
           'downgreen',
           'downpink',
           'downpurple',
           'downwhite',
           'downyellow',
           'upblack',
           'upblue',
           'upgreen',
           'upgray',
           'uppurple',
           'upred',
           'upwhite',
           'upyellow',
           'clothes',
           'down',
           'up',
           'hair',
           'hat',
           'gender']
    
    test_label=['age',
           'backpack',
           'bag',
           'handbag',
           'clothes',
           'down',
           'up',
           'hair',
           'hat',
           'gender',
           'upblack',
           'upwhite',
           'upred',
           'uppurple',
           'upyellow',
           'upgray',
           'upblue',
           'upgreen',
           'downblack',
           'downwhite',
           'downpink',
           'downpurple',
           'downyellow',
           'downgray',
           'downblue',
           'downgreen',
           'downbrown'
           ]
    
    train_person_id = []
    for personid in train:
        train_person_id.append(personid)
    train_person_id.sort(key=int)

    test_person_id = []
    for personid in test:
        test_person_id.append(personid)
    test_person_id.sort(key=int)
    test_person_id.remove('-1')
    test_person_id.remove('0000')

    f = scipy.io.loadmat(os.path.join(dataset_dir,dataset_name,'market_attribute.mat'))

    test_attribute = {}
    train_attribute = {}
    for test_train in range(len(f['market_attribute'][0][0])):
        if test_train == 0:
            id_list_name = 'test_person_id'
            group_name = 'test_attribute'
        else:
            id_list_name = 'train_person_id'
            group_name = 'train_attribute'
        for attribute_id in range(len(f['market_attribute'][0][0][test_train][0][0])):
            if isinstance(f['market_attribute'][0][0][test_train][0][0][attribute_id][0][0], np.ndarray):
                continue
            for person_id in range(len(f['market_attribute'][0][0][test_train][0][0][attribute_id][0])):
                id = locals()[id_list_name][person_id]
                if id not in locals()[group_name]:
                    locals()[group_name][id]=[]
                locals()[group_name][id].append(f['market_attribute'][0][0][test_train][0][0][attribute_id][0][person_id])
    
    unified_train_atr = {}
    for k,v in train_attribute.items():
        temp_atr = [0]*len(test_label)
        for i in range(len(test_label)):
            temp_atr[i]=v[train_label.index(test_label[i])]
        unified_train_atr[k] = temp_atr
    
    return unified_train_atr, test_attribute, test_label


def import_Market1501Attribute_binary(dataset_dir):
    train_market_attr, test_market_attr, label = import_Market1501Attribute(dataset_dir)
    
    for id in train_market_attr:
        train_market_attr[id][:] = [x - 1 for x in train_market_attr[id]]
        if train_market_attr[id][0] == 0:
            train_market_attr[id].pop(0)
            train_market_attr[id].insert(0, 1)
            train_market_attr[id].insert(1, 0)
            train_market_attr[id].insert(2, 0)
            train_market_attr[id].insert(3, 0)
        elif train_market_attr[id][0] == 1:
            train_market_attr[id].pop(0)
            train_market_attr[id].insert(0, 0)
            train_market_attr[id].insert(1, 1)
            train_market_attr[id].insert(2, 0)
            train_market_attr[id].insert(3, 0)
        elif train_market_attr[id][0] == 2:
            train_market_attr[id].pop(0)
            train_market_attr[id].insert(0, 0)
            train_market_attr[id].insert(1, 0)
            train_market_attr[id].insert(2, 1)
            train_market_attr[id].insert(3, 0)
        elif train_market_attr[id][0] == 3:
            train_market_attr[id].pop(0)
            train_market_attr[id].insert(0, 0)
            train_market_attr[id].insert(1, 0)
            train_market_attr[id].insert(2, 0)
            train_market_attr[id].insert(3, 1)

    for id in test_market_attr:
        test_market_attr[id][:] = [x - 1 for x in test_market_attr[id]]
        if test_market_attr[id][0] == 0:
            test_market_attr[id].pop(0)
            test_market_attr[id].insert(0, 1)
            test_market_attr[id].insert(1, 0)
            test_market_attr[id].insert(2, 0)
            test_market_attr[id].insert(3, 0)
        elif test_market_attr[id][0] == 1:
            test_market_attr[id].pop(0)
            test_market_attr[id].insert(0, 0)
            test_market_attr[id].insert(1, 1)
            test_market_attr[id].insert(2, 0)
            test_market_attr[id].insert(3, 0)
        elif test_market_attr[id][0] == 2:
            test_market_attr[id].pop(0)
            test_market_attr[id].insert(0, 0)
            test_market_attr[id].insert(1, 0)
            test_market_attr[id].insert(2, 1)
            test_market_attr[id].insert(3, 0)
        elif test_market_attr[id][0] == 3:
            test_market_attr[id].pop(0)
            test_market_attr[id].insert(0, 0)
            test_market_attr[id].insert(1, 0)
            test_market_attr[id].insert(2, 0)
            test_market_attr[id].insert(3, 1)

    label.pop(0)
    label.insert(0,'young')
    label.insert(1,'teenager')
    label.insert(2,'adult')
    label.insert(3,'old')
    
    return train_market_attr, test_market_attr, label