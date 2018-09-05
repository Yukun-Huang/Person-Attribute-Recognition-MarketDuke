import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchvision import models

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)



class ResNet50_nFC_softmax(nn.Module):
    def __init__(self,class_num, id_num, **kwargs):
        super(ResNet50_nFC_softmax, self).__init__()
        self.model_name = 'resnet50_nfc_softmax'
        self.class_num = class_num
        self.id_num = id_num
        
        model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.features = model_ft
        self.num_ftrs = 2048
        num_bottleneck = 512
        
        for c in range(self.class_num+1):
            if c == self.class_num:
                self.__setattr__('class_%d' % c,
                nn.Sequential(nn.Linear(self.num_ftrs,num_bottleneck), 
                              nn.BatchNorm1d(num_bottleneck),
                              nn.LeakyReLU(0.1),
                              nn.Dropout(p=0.5),
                              nn.Linear(num_bottleneck, self.id_num)))    
            else:
                self.__setattr__('class_%d' % c,
                nn.Sequential(nn.Linear(self.num_ftrs,num_bottleneck), 
                              nn.BatchNorm1d(num_bottleneck),
                              nn.LeakyReLU(0.1),
                              nn.Dropout(p=0.5),
                              nn.Linear(num_bottleneck, 2)))      

    def forward(self, x):
        x = self.features(x)
        return (self.__getattr__('class_%d' % c)(x) 
            for c in range(self.class_num+1)), x
