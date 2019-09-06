import os
import argparse
import scipy.io
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datafolder.folder import Test_Dataset
from net import *
import warnings
warnings.filterwarnings("ignore")


######################################################################
# Settings
# ---------
use_gpu = True
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


######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--data-path', default='/home/xxx/reid/', type=str, help='path to the dataset')
parser.add_argument('--dataset', default='duke', type=str, help='dataset')
parser.add_argument('--model', default='resnet50', type=str, help='model')
parser.add_argument('--batch-size', default=50, type=int, help='batch size')
parser.add_argument('--num-epoch', default=60, type=int, help='num of epoch')
parser.add_argument('--num-workers', default=2, type=int, help='num_workers')
parser.add_argument('--which-epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--print-table',action='store_true', help='print results with table format')
args = parser.parse_args()

assert args.dataset in dataset_dict.keys()
assert args.model in model_dict.keys()

batch_size = args.batch_size
data_dir = args.data_path
model_dir = os.path.join('./checkpoints', args.dataset, args.model)
result_dir = os.path.join('./result', args.dataset, args.model)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)


######################################################################
# Function
# ---------
def load_network(network):
    save_path = os.path.join(model_dir,'net_%s.pth'%args.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


def get_dataloader():
    image_datasets = {}
    image_datasets['gallery'] = Test_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                             query_gallery='gallery')
    image_datasets['query'] = Test_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                           query_gallery='query')
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers)
                  for x in ['gallery', 'query']}
    return dataloaders


######################################################################
# Load Data
# ---------
# Note that we only perform evaluation on gallery set.
test_loader = get_dataloader()['gallery']

attribute_list = test_loader.dataset.labels()
num_label = len(attribute_list)
num_sample = len(test_loader.dataset)


######################################################################
# Model
# ---------
model = model_dict[args.model](num_label)
model = load_network(model)
if use_gpu:
    model = model.cuda()
model.train(False)  # Set model to evaluate mode


######################################################################
# Testing
# ---------
preds_tensor = np.empty(shape=[0, num_label], dtype=np.byte)   # shape = (num_sample, num_label)
labels_tensor = np.empty(shape=[0, num_label], dtype=np.byte)   # shape = (num_sample, num_label)

# Iterate over data.
for count, (images, labels, ids, file_name) in enumerate(test_loader):
    # move input to GPU
    if use_gpu:
        images = images.cuda()
    # forward
    outputs = model(images)
    preds = torch.gt(outputs, torch.ones_like(outputs)/2 )
    # transform to numpy format
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    # append
    preds_tensor = np.append(preds_tensor, preds, axis=0)
    labels_tensor = np.append(labels_tensor, labels, axis=0)
    # print info
    if count*batch_size % 5000 == 0:
        print('Step: {}/{}'.format(count*batch_size, num_sample))

# Evaluation.
accuracy_list = []
precision_list = []
recall_list = []
f1_score_list = []
for i, name in enumerate(attribute_list):
    y_true, y_pred = labels_tensor[:, i], preds_tensor[:, i]
    accuracy_list.append(accuracy_score(y_true, y_pred))
    precision_list.append(precision_score(y_true, y_pred, average='binary'))
    recall_list.append(recall_score(y_true, y_pred, average='binary'))
    f1_score_list.append(f1_score(y_true, y_pred, average='binary'))


######################################################################
# Print
# ---------
print("\n"
      "Some attributes may not have a positive (or negative) sample,"
      "therefore 'UndefinedMetricWarning' is raised when calculating Precision, Recall and F-score."
      "\n")

if args.print_table:
    from prettytable import PrettyTable
    table = PrettyTable(['attribute', 'accuracy', 'precision', 'recall', 'f1 score'])
    for i, name in enumerate(attribute_list):
        y_true, y_pred = labels_tensor[:, i], preds_tensor[:, i]
        precision_list.append(precision_score(y_true, y_pred, average='binary'))
        recall_list.append(recall_score(y_true, y_pred, average='binary'))
        f1_score_list.append(f1_score(y_true, y_pred, average='binary'))
        table.add_row([name,
               '%.3f' % accuracy_list[i],
               '%.3f' % precision_list[i],
               '%.3f' % recall_list[i],
               '%.3f' % f1_score_list[i],
               ])
    print(table)

average_acc = np.mean(accuracy_list)
average_precision = np.mean(precision_list)
average_recall = np.mean(recall_list)
average_f1score = np.mean(f1_score_list)

print('Average accuracy: {:.4f}'.format(average_acc))
# print('Average precision: {:.4f}'.format(average_precision))
# print('Average recall: {:.4f}'.format(average_recall))
print('Average f1 score: {:.4f}'.format(average_f1score))

# Save results.
result = {
    'average_acc'       :   average_acc,
    'average_f1score'   :   average_f1score,
    'accuracy_list'     :   accuracy_list,
    'precision_list'    :   precision_list,
    'recall_list'       :   recall_list,
    'f1_score_list'     :   f1_score_list,
}
scipy.io.savemat(os.path.join(result_dir, 'acc.mat'), result)


