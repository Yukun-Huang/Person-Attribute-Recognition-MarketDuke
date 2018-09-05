import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from .reid_dataset import import_MarketDuke_nodistractors
from .reid_dataset import import_Market1501Attribute_binary
from .reid_dataset import import_DukeMTMCAttribute_binary


class Train_Dataset(data.Dataset):

    def __init__(self, data_dir, dataset_name, transforms=None, train_val='train' ):

        train, query, gallery = import_MarketDuke_nodistractors(data_dir, dataset_name)

        if dataset_name == 'Market-1501':
            train_attr, test_attr, self.label = import_Market1501Attribute_binary(data_dir)
        elif dataset_name == 'DukeMTMC-reID':
            train_attr, test_attr, self.label = import_DukeMTMCAttribute_binary(data_dir)
        else:
            print('Input should only be Market1501 or DukeMTMC')

        self.num_ids = len(train['ids'])
        self.num_labels = len(self.label)

        # distribution:每个属性的正样本占比
        distribution = np.zeros(self.num_labels)
        for k, v in train_attr.items():
            distribution += np.array(v)
        self.distribution = distribution / len(train_attr)

        if train_val == 'train':
            self.train_data = train['data']
            self.train_ids = train['ids']
            self.train_attr = train_attr
        elif train_val == 'query':
            self.train_data = query['data']
            self.train_ids = query['ids']
            self.train_attr = test_attr
        elif train_val == 'gallery':
            self.train_data = gallery['data']
            self.train_ids = gallery['ids']
            self.train_attr = test_attr
        else:
            print('Input should only be train or val')

        self.num_ids = len(self.train_ids)

        if transforms is None:
            if train_val == 'train':
                self.transforms = T.Compose([
                    T.Resize(size=(288, 144)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(size=(288, 144)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.train_data[index][0]
        i = self.train_data[index][1]
        id = self.train_data[index][2]
        cam = self.train_data[index][3]
        label = np.asarray(self.train_attr[id])
        data = Image.open(img_path)
        data = self.transforms(data)
        name = self.train_data[index][4]
        return data, i, label, id, cam, name

    def __len__(self):
        return len(self.train_data)

    def num_label(self):
        return self.num_labels

    def num_id(self):
        return self.num_ids

    def labels(self):
        return self.label



class Test_Dataset(data.Dataset):
    def __init__(self, data_dir, dataset_name, transforms=None, query_gallery='query' ):
        train, query, gallery = import_MarketDuke_nodistractors(data_dir, dataset_name)

        if dataset_name == 'Market-1501':
            self.train_attr, self.test_attr, self.label = import_Market1501Attribute_binary(data_dir)
        elif dataset_name == 'DukeMTMC-reID':
            self.train_attr, self.test_attr, self.label = import_DukeMTMCAttribute_binary(data_dir)
        else:
            print('Input should only be Market1501 or DukeMTMC')

        if query_gallery == 'query':
            self.test_data = query['data']
            self.test_ids = query['ids']
        elif query_gallery == 'gallery':
            self.test_data = gallery['data']
            self.test_ids = gallery['ids']
        elif query_gallery == 'all':
            self.test_data = gallery['data'] + query['data']
            self.test_ids = gallery['ids']
        else:
            print('Input shoud only be query or gallery;')

        if transforms is None:
            self.transforms = T.Compose([
                T.Resize(size=(288, 144)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.test_data[index][0]
        id = self.test_data[index][2]
        label = np.asarray(self.test_attr[id])
        data = Image.open(img_path)
        data = self.transforms(data)
        name = self.test_data[index][4]
        return data, label, id, name

    def __len__(self):
        return len(self.test_data)

    def labels(self):
        return self.label