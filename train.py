# !/usr/local/bin/python3
import os
import time
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datafolder.folder import Train_Dataset
from net import get_model

######################################################################
# Settings
# --------
use_gpu = True
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}


######################################################################
# Argument
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data-path', default='/path/to/dataset', type=str, help='path to the dataset')
parser.add_argument('--dataset', default='market', type=str, help='dataset: market, duke')
parser.add_argument('--backbone', default='resnet50', type=str, help='backbone: resnet50, resnet34, resnet18, densenet121')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--num-epoch', default=60, type=int, help='num of epoch')
parser.add_argument('--num-workers', default=2, type=int, help='num_workers')
parser.add_argument('--use-id', action='store_true', help='use identity loss')
parser.add_argument('--lamba', default=1.0, type=float, help='weight of id loss')
args = parser.parse_args()

assert args.dataset in ['market', 'duke']
assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

dataset_name = dataset_dict[args.dataset]
model_name = '{}_nfc_id'.format(args.backbone) if args.use_id else '{}_nfc'.format(args.backbone)
data_dir = args.data_path
model_dir = os.path.join('./checkpoints', args.dataset, model_name)

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)


######################################################################
# Function
# --------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(model_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if use_gpu:
        network.cuda()
    print('Save model to {}'.format(save_path))


######################################################################
# Draw Curve
#-----------
x_epoch = []
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join(model_dir, 'train.jpg'))


######################################################################
# DataLoader
# ---------
image_datasets = {}
image_datasets['train'] = Train_Dataset(data_dir, dataset_name=dataset_name, train_val='train')
image_datasets['val'] = Train_Dataset(data_dir, dataset_name=dataset_name, train_val='query')
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers, drop_last=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# images, indices, labels, ids, cams, names = next(iter(dataloaders['train']))

num_label = image_datasets['train'].num_label()
num_id = image_datasets['train'].num_id()
labels_list = image_datasets['train'].labels()


######################################################################
# Model and Optimizer
# ------------------
model = get_model(model_name, num_label, args.use_id, num_id=num_id)
if use_gpu:
    model = model.cuda()

# loss
criterion_bce = nn.BCELoss()
criterion_ce = nn.CrossEntropyLoss()

# optimizer
ignored_params = list(map(id, model.features.parameters()))
classifier_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer = torch.optim.SGD([
            {'params': model.features.parameters(), 'lr': 0.01},
            {'params': classifier_params, 'lr': 0.1},
        ], momentum=0.9, weight_decay=5e-4, nesterov=True)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)


######################################################################
# Training the model
# ------------------
def train_model(model, optimizer, scheduler, num_epochs):
    since = time.time()

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for count, (images, indices, labels, ids, cams, names) in enumerate(dataloaders[phase]):
                # get the inputs
                labels = labels.float()
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                    indices = indices.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if not args.use_id:
                    pred_label = model(images)
                    total_loss = criterion_bce(pred_label, labels)
                else:
                    pred_label, pred_id = model(images)
                    label_loss = criterion_bce(pred_label, labels)
                    id_loss = criterion_ce(pred_id, indices)
                    total_loss = label_loss + args.lamba * id_loss

                # backward + optimize only if in training phase
                if phase == 'train':
                    total_loss.backward()
                    optimizer.step()

                preds = torch.gt(pred_label, torch.ones_like(pred_label)/2 )
                # statistics
                running_loss += total_loss.item()
                running_corrects += torch.sum(preds == labels.byte()).item() / num_label
                if count % 100 == 0:
                    if not args.use_id:
                        print('step: ({}/{})  |  label loss: {:.4f}'.format(
                            count*args.batch_size, dataset_sizes[phase], total_loss.item()))
                    else:
                        print('step: ({}/{})  |  label loss: {:.4f}  |  id loss: {:.4f}'.format(
                            count*args.batch_size, dataset_sizes[phase], label_loss.item(), id_loss.item()))

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 0:
                    save_network(model, epoch)
                draw_curve(epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')


######################################################################
# Main
# -----
train_model(model, optimizer, exp_lr_scheduler, num_epochs=args.num_epoch)
