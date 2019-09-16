import os
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import *


######################################################################
# Settings
# ---------
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
model_dict = {
    'resnet18'  :  ResNet18_nFC,
    'resnet34'  :  ResNet34_nFC,
    'resnet50'  :  ResNet50_nFC,
    'densenet'  :  DenseNet121_nFC,
    'resnet50_softmax'  :  ResNet50_nFC_softmax,
}
num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='Path to test image')
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--model', default='resnet50', type=str, help='model')
args = parser.parse_args()


######################################################################
# Model and Data
# ---------
def load_network(network):
    save_path = os.path.join('./checkpoints', args.dataset, args.model, 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    return network

def load_image(path):
    src = Image.open(path)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src

model = model_dict[args.model](num_cls_dict[args.dataset])
model = load_network(model)
model.eval()

src = load_image(args.image_path)

######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        with open('./doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('./doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred):
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]]:
                print('{}: {}'.format(name, chooce[pred[idx]]))


out = model.forward(src)
pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5

Dec = predict_decoder(args.dataset)
Dec.decode(pred)

