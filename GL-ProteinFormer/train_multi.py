import sys
# sys.path.append('models')
import os
import argparse

import timm

import numpy as np
import random
from PIL import Image
import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from torch.utils.data import DataLoader
import torch.utils.data as data

from models.resnet import get_resnet50
from models.resnet_pre import resnet50
from models.VGG import vgg16_bn
from models.densenet import densenet169
from models.VIT import ViT
from models.vit import VisionTransformer, vit_base_patch16
from models.vit_stem import vit_base_patch16_stem
from models.vit_stem_crop import vit_base_patch16_stem_crop
from models.vit_small import ViTS
from models.cvt import get_cls_cvt
from models.cvt_res import get_cls_cvt_res
from models.procls.cls_m import CLSModel, STModel, MultiScaleSTModel, CLSModel_Pre
from models.gl_ptformer.cls_model import GL_PTFormer
# import temp
from models.swin import swin_tiny, swin_base

from datasets.hpa import HpaDataset
from datasets.hpa_multi import HpaMultiDataset
from datasets.ProteinLoc import ProLocDataset

from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torchvision.transforms as T
import torchvision

from torch.utils.tensorboard import SummaryWriter
# import pyll




parser = argparse.ArgumentParser()
parser.add_argument('--save_root', default='checkpoints/debug', help="name your save_root")
parser.add_argument('--load_path', default=None, help="name your load_path")
parser.add_argument('--pretrained_feature', default=False, type=str,  help='use pretrained feature')
parser.add_argument('--loadckpt', default=False, type=str,  help='use pretrained feature')
args = parser.parse_args()
torch.manual_seed(42)
np.random.seed(42)
print('args.save_root', args.save_root)
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
if not os.path.isdir(args.save_root):
    os.makedirs(args.save_root)

# data_directory_path = "/data2/zhangjw/help/dataset/setall"
# train_file = "/data2/zhangjw/help/dataset/setall/labels_train_HE_multiclass.txt"
# valid_file = "/data2/zhangjw/help/dataset/setall/labels_val_HE_multiclass.txt"
# test_file = "/data2/zhangjw/help/dataset/setall/labels_test_HE_multiclass.txt"

# pretrained_dir = '/data2/model_zoo/vit_base_patch16_224.pth.tar'
# pretrained_dir = None

# load_PATH = '/data2/zhangjw/help/checkpoints/resnet/pretrain_lr5e5/30_acc_91.pth'


# NUM_CLASSES = 13

SUM_FREQ = 100




start_epoch = 0

def load_checkpoint(model, checkpoint_PATH, optimizer, resume=False):
    print('loading checkpoint!')
    start_epoch = 0
    if checkpoint_PATH != None:
        if resume:
            model_CKPT = torch.load(checkpoint_PATH, map_location=lambda storage, loc: storage.cuda())
            state_dict = dict()
            for k, v in model_CKPT['state_dict'].items():
                # new_k = k.replace('module.','')
                # new_k = 'module.' + k
                new_k = k
                # print(new_k)
                # if new_k != 'conv1.weight' and new_k != 'fc.weight' and new_k != 'fc.bias':
                # if new_k != 'module.patch_embed.proj.weight':
                state_dict[new_k] = v  
            model.load_state_dict(state_dict)
            # model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
            print('loading checkpoint!')
            optimizer.load_state_dict(model_CKPT['optimizer'])
            print('loading optimizer!')
            start_epoch = model_CKPT['epoch']
        else:
            model_CKPT = torch.load(checkpoint_PATH)
            model.load_state_dict(model_CKPT['state_dict'])
            print('loading checkpoint!')
    print('start_epoch', start_epoch)
    return start_epoch




# DEVICE = 'cuda:7'
gpus = [0,1]
# gpus = [0,1,2,3,4,5,6,7]
# gpus = [6, 7]

# lr_value = 0.0001
# lr_value = 1e-3
# lr_value = 5e-4
# lr_value = 1e-4
# lr_value = 5e-5
# lr_value = 1e-5
lr_value = 5e-6
# lr_value = 1e-7
# lr_value = 0.001
# lr_value = 0.005
# lr_value = 0.00005
# lr_value = 0.00001
# lr_value = 0.01
total_epochs = 10000
# batch_size = 8
# batch_size = 256
# batch_size = 16
batch_size = 64
# batch_size = 32
# batch_size = 1
test_batch_size = 1

VAL_FREQ = 10
TEST_FREQ = 50

MEAN = [0, 0, 0, 0]
STD = [1, 1, 1, 1]
# CROP_SIZE = (128, 128)
# CROP_SIZE = (256, 256)
# CROP_SIZE = (512, 512)
# CROP_SIZE = (1024, 1024)

# train_transforms = T.Compose([
#         torchvision.transforms.ToPILImage('CMYK'),
#         torchvision.transforms.RandomCrop(CROP_SIZE),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=MEAN, std=STD),
#         # pyll.transforms.NormalizeByImage()
#       ])
# train_transforms = T.Compose([
#         torchvision.transforms.ToPILImage('CMYK'),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=MEAN, std=STD),
#         # pyll.transforms.NormalizeByImage()
#       ])


# eval_transforms = T.Compose([
#         torchvision.transforms.ToPILImage('CMYK'),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=MEAN, std=STD),
#         # pyll.transforms.NormalizeByImage()
#       ])

print('loading data')
# train_dataset = ProLocDataset(data_directory_path=data_directory_path, label_file=train_file, transforms=train_transforms, num_classes=13)
# # train_dataset = ProLocDataset(data_directory_path=data_directory_path, label_file=train_file, transforms=train_transforms, num_classes=13, patching=True, patch_size=CROP_SIZE[0])
# print("training data count: %d" % len(train_dataset))
# valid_dataset = ProLocDataset(data_directory_path=data_directory_path, label_file=valid_file, transforms=eval_transforms, num_classes=13, patching=True, patch_size=CROP_SIZE[0])
# print("valid data count: %d" % len(valid_dataset))
# test_dataset = ProLocDataset(data_directory_path=data_directory_path, label_file=test_file, transforms=eval_transforms, num_classes=13, patching=True, patch_size=CROP_SIZE[0])
# print("testing data count: %d" % len(test_dataset))


train_annotation_file = 'filenames/multi_cls/train.txt'
val_annotation_file = 'filenames/multi_cls/val.txt'
test_annotation_file = 'filenames/multi_cls/test.txt'

data_root = 'data_root'
train_dataset = HpaMultiDataset(split='train', root=data_root, annotation=train_annotation_file)
print("training data count: %d" % len(train_dataset))
valid_dataset = HpaMultiDataset(split='test', root=data_root, annotation=val_annotation_file)
print("testing data count: %d" % len(valid_dataset))
test_dataset = HpaMultiDataset(split='test', root=data_root, annotation=test_annotation_file)
print("valid data count: %d" % len(test_dataset))

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, 
        pin_memory=False, shuffle=True, num_workers=0, drop_last=True)

eval_loader = data.DataLoader(valid_dataset, batch_size=test_batch_size, 
        pin_memory=False, shuffle=False, num_workers=0, drop_last=False)

test_loader = data.DataLoader(test_dataset, batch_size=test_batch_size, 
        pin_memory=False, shuffle=False, num_workers=0, drop_last=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



model = nn.DataParallel(GL_PTFormer(n_classes=9), device_ids=gpus)

print("Parameter Count: %d" % count_parameters(model))

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_value)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)



activate_list = [n for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
print('len_activate_list', len(activate_list))
# print('activate_list', activate_list)
if args.pretrained_feature:
    print('load_pretrained_model_from{}'.format(args.pretrained_feature))
    pre_feat_state_dict = torch.load(args.pretrained_feature, map_location=lambda storage, loc: storage.cuda())
    if 'model' in pre_feat_state_dict.keys():
        pre_feat_state_dict = pre_feat_state_dict['model']
    elif 'state_dict' in pre_feat_state_dict.keys():
        pre_feat_state_dict = pre_feat_state_dict['state_dict']
    feat_dict = model.module.backbone.state_dict()
    pre_feat_state_dict = {k.replace('module.',''): v for k, v in pre_feat_state_dict.items()}
    pre_feat_state_dict = {k.replace('backbone.',''): v for k, v in pre_feat_state_dict.items()}
    pre_dict = {k: v for k, v in pre_feat_state_dict.items() if k in feat_dict}
    # print('matching keys', set(state_dict.keys()) & set(pre_dict.keys()))
    print('feature unexpceted keys', set(pre_feat_state_dict.keys()) - set(feat_dict.keys()))
    print('feature unloaded keys', set(feat_dict.keys()) - set(pre_feat_state_dict.keys()))
    print('feature length of unloaded keys = {}'.format(len(set(feat_dict.keys()) - set(pre_feat_state_dict.keys()))))
    unloaded_list = list(set(feat_dict.keys()) - set(pre_feat_state_dict.keys()))
    un_key_list = []
    for i in range(len(unloaded_list)):
        un_key_list.append('module.backbone.' + unloaded_list[i])
    print('len of strange keys = {}'.format(len(set(un_key_list) - set(activate_list))))
    print('strange keys', set(un_key_list) - set(activate_list))
    # print('matching keys', set(dct0.keys()) & set(dct1.keys()))
    # print('matching keys', set(dct0.keys()) & set(dct1.keys()))
    # model.module.load_pretrained_feature(feat_state_dict)
    feat_dict.update(pre_dict)
    model.module.backbone.load_state_dict(feat_dict)
    del pre_dict
    del feat_dict
    del pre_feat_state_dict
elif args.loadckpt:
    # # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=lambda storage, loc: storage.cuda())
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['state_dict'].items() if k in model_dict}
    print('what happened ?')
    print('matching keys', set(state_dict.keys()) & set(pre_dict.keys()))
    print('unexpceted keys', set(state_dict['state_dict'].keys()) - set(model_dict.keys()))
    print('unloaded keys', set(model_dict.keys()) - set(state_dict['state_dict'].keys()))
    print('length of unloaded keys = {}'.format(len(set(model_dict.keys()) - set(state_dict['state_dict'].keys()))))
    # print('matching keys', set(dct0.keys()) & set(dct1.keys()))
    # print('matching keys', set(dct0.keys()) & set(dct1.keys()))
    model_dict.update(pre_dict) 
    model.load_state_dict(model_dict, strict=True)
    del pre_dict
    del model_dict
    del state_dict

model.cuda()



total_steps = 0


def calculate_metrics(label_bins, pred_bins, name):
    acc = accuracy_score(y_true=label_bins.cpu(), y_pred=pred_bins.cpu())
    pre = precision_score(y_true=label_bins.cpu(), y_pred=pred_bins.cpu(), average=None)
    rec = recall_score(y_true=label_bins.cpu(), y_pred=pred_bins.cpu(), average=None)
    f1 = f1_score(y_true=label_bins.cpu(), y_pred=pred_bins.cpu(), average=None)
    # metrics = dict()
    # metrics.update(accuracy_score=acc)
    # metrics.update(precision_score=pre)
    # metrics.update(recall_score=rec)
    # metrics.update(f1_score=f1)
    # metrics = {name+'accuracy_score': acc, name+'precision_score_mean': pre.mean(), 
    #     name+'recall_score_mean': rec.mean(), name+'f1_score_mean': f1.mean(),
    #     name+'precision_score': pre, name+'recall_score': rec, name+'f1_score': f1}
    metrics = {name+'accuracy_score': acc, name+'precision_score_mean': pre.mean(), 
        name+'recall_score_mean': rec.mean(), name+'f1_score_mean': f1.mean()}
    return metrics


def evaluate(args=None, epoch=-1, dataset=None, tgt_bins=None):
    print('evaluate epoch{}'.format(epoch))
    model.eval()
    total_num_correct = 0
    total_num = 0
    pred_bins = []
    label_bins = []
    for iteration, data in enumerate(dataset):
        image, label = data['input'], data['target']
        image = image.cuda()
        label = label.cuda()
        # B, N, _, H, W = image.shape
        # image = image.view(B * N, 4, H, W)
        with torch.no_grad():
            logits, feat = model(image)
            # logits = logits.view(B, N, -1)
            # logits = logits.mean(dim=1)
            # print('logits.shape', logits.shape)
        
        # pred_bins.append(logits)
        pred_bins.append(torch.sigmoid(logits))
        label_bins.append(label)
        print('Evaluate:[{}/{}]\t'.format(iteration, len(dataset)))
    
    pred_bins = torch.cat(pred_bins, dim=0)
    label_bins = torch.cat(label_bins, dim=0)
    # pred_bins = (pred_bins == pred_bins.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
    # label_bins = (label_bins == label_bins.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
    print('pred_bins', pred_bins.shape)
    print('label_bins', label_bins.shape)
    print('label_bins', label_bins)
    results = dict()
    # print('pred_bins', pred_bins)
    # pred_bins_5e1 = (pred_bins + 0.5).astype(int)
    pred_bins_threshold = (pred_bins >= 0.1).to(dtype=torch.int32)
    metrics = calculate_metrics(label_bins, pred_bins_threshold, '0.1_')
    results.update(metrics)
    pred_bins_threshold = (pred_bins >= 0.3).to(dtype=torch.int32)
    metrics = calculate_metrics(label_bins, pred_bins_threshold, '0.3_')
    results.update(metrics)
    pred_bins_threshold = (pred_bins >= 0.5).to(dtype=torch.int32)
    metrics = calculate_metrics(label_bins, pred_bins_threshold, '0.5_')
    results.update(metrics)
    pred_bins_threshold = (pred_bins >= 0.7).to(dtype=torch.int32)
    metrics = calculate_metrics(label_bins, pred_bins_threshold, '0.7_')
    results.update(metrics)
    pred_bins_threshold = (pred_bins >= 0.9).to(dtype=torch.int32)
    metrics = calculate_metrics(label_bins, pred_bins_threshold, '0.9_')
    results.update(metrics)
    print('results', results)
    return results

# start_epoch = 0
def train(args=None):
    for epoch in range(start_epoch+1, total_epochs+1):
        model.train()
        # model.freeze_bn()
        print("epoch: {}, lr: {:.6f}".format(epoch, optimizer.param_groups[0]['lr']))
        for iteration, data in enumerate(train_loader):
            image, label = data['input'], data['target']  # batch_size * 3 * 224 * 224   /  batch+size * 1
            image = image.cuda()
            label = label.cuda()
            logits, features = model(image) # batch_size * 1
            # loss = F.cross_entropy(logits, label)
            logits = torch.sigmoid(logits).float()
            label = label.float()
            loss = F.binary_cross_entropy(logits, label)
            print('Train:[{}/{}][{}/{}] , loss:{} \t'.format(epoch, total_epochs, iteration, len(train_loader), loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        
        if epoch % VAL_FREQ == 0:
            results = evaluate(args, epoch, eval_loader)
            save_NAME = 'val_epoch_%d.pth' % (epoch)
            save_PATH = os.path.join(args.save_root, save_NAME)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_PATH)

        if epoch % TEST_FREQ == 0:
            results = evaluate(args, epoch, test_loader)
            save_NAME = 'test_epoch_%d.pth' % (epoch)
            save_PATH = os.path.join(args.save_root, save_NAME)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_PATH)



if __name__ == '__main__':
    train(args)