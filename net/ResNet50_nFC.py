import torch
from torch import nn
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



# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, num_bottleneck = 512):
        super(ClassBlock, self).__init__()


        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]
        add_block += [nn.Linear(num_bottleneck, 1)]
        add_block += [nn.Sigmoid()]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        self.classifier = add_block

    def forward(self, x):
        x = self.classifier(x)
        return x


class ResNet50_nFC(nn.Module):
    def __init__(self, class_num):
        super(ResNet50_nFC, self).__init__()
        self.model_name = 'resnet50_nfc'
        self.class_num = class_num

        model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.features = model_ft
        self.num_ftrs = 2048
        num_bottleneck = 512

        for c in range(self.class_num):
            self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, num_bottleneck) )

    def forward(self, x):
        x = self.features(x)
        for c in range(self.class_num):
            if c == 0:
                pred = self.__getattr__('class_%d' % c)(x)
            else:
                pred = torch.cat((pred, self.__getattr__('class_%d' % c)(x) ), dim=1)
        return pred