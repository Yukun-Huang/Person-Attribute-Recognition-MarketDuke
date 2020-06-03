import torch
from torch import nn
from torch.nn import init
from torchvision import models
from net.utils import ClassBlock
from torch.nn import functional as F


class Backbone_nFC(nn.Module):
    def __init__(self, class_num, model_name='resnet50_nfc'):
        super(Backbone_nFC, self).__init__()
        self.model_name = model_name
        self.backbone_name = model_name.split('_')[0]
        self.class_num = class_num

        model_ft = getattr(models, self.backbone_name)(pretrained=True)
        if 'resnet' in self.backbone_name:
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft
            self.num_ftrs = 2048
        elif 'densenet' in self.backbone_name:
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft.features
            self.num_ftrs = 1024
        else:
            raise NotImplementedError
        
        for c in range(self.class_num):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.num_ftrs, class_num=1, activ='sigmoid') )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(self.class_num)]
        pred_label = torch.cat(pred_label, dim=1)
        return pred_label


class Backbone_nFC_Id(nn.Module):
    def __init__(self, class_num, id_num, model_name='resnet50_nfc_id'):
        super(Backbone_nFC_Id, self).__init__()
        self.model_name = model_name
        self.backbone_name = model_name.split('_')[0]
        self.class_num = class_num
        self.id_num = id_num
        
        model_ft = getattr(models, self.backbone_name)(pretrained=True)
        if 'resnet' in self.backbone_name:
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft
            self.num_ftrs = 2048
        elif 'densenet' in self.backbone_name:
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft.features
            self.num_ftrs = 1024
        else:
            raise NotImplementedError
        
        for c in range(self.class_num+1):
            if c == self.class_num:
                self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, class_num=self.id_num, activ='none'))
            else:
                self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, class_num=1, activ='sigmoid'))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(self.class_num)]
        pred_label = torch.cat(pred_label, dim=1)
        pred_id = self.__getattr__('class_%d' % self.class_num)(x)
        return pred_label, pred_id

