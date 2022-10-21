from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
import models.resnet_imagenet as resnet_imagenet

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.svhn import get_svhn_dataloaders # kd-Huan: add svhn dataset
from dataset.tinyimagenet import get_tinyimagenet_dataloaders # kd-Huan: add tinyimagenet dataset
from dataset.imagenet import get_imagenet_dataloader

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import validate

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg8_3neurons', 'vgg11', 'vgg13', 'vgg13_3neurons', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'MobileNetV2_0_25', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_0_3', 'ResNet50', 'resnet18'])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'svhn', 'tinyimagenet', 'imagenet'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    # --- kd-Huan
    parser.add_argument('--project_name', type=str, default="")
    parser.add_argument('--CodeID', type=str, default="")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--screen_print', action="store_true")
    parser.add_argument('--resume_ExpID', type=str)
    parser.add_argument('--note', type=str)
    parser.add_argument('--save_img_interval', type=int, default=200)
    parser.add_argument('--npy_set', type=str, default='')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--lw_selfkd', type=float, default=0)
    parser.add_argument('--epoch_factor', type=float, default=0)
    parser.add_argument('--use_DA', type=str, default='11', help='first 1 indicates using rand crop; second 1 indicates using horizontal flip')
    parser.add_argument('--no_DA', action='store_true', help='maintain back-compatibility, deprecated. Use --use_DA')
    parser.add_argument('--branch_layer', type=str, default='[]')
    parser.add_argument('--pretrained', type=str, help='pretrained model')
    parser.add_argument('--transfer', action="store_true")
    opt = parser.parse_args()
    opt.branch_layer = strlist_to_list(opt.branch_layer, int)
    # ---
    
    # set different learning rate from these 4 models
    if opt.dataset in ['cifar100', 'tinyimagenet']:
        if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
            opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # --- kd-Huan: for easier scale of epochs like 2xtime, 2.5xtime
    if opt.epoch_factor:
        opt.lr_decay_epochs = [int(x * opt.epoch_factor) for x in opt.lr_decay_epochs]
        opt.epochs = int(opt.epochs * opt.epoch_factor)
    # ---

    if opt.no_DA: # maintain back-compatibility
        opt.use_DA = '00'
    return opt

# --- kd-Huan
from logger import Logger
from utils import Timer, strlist_to_list, smart_weights_load, cal_acc
from branch import BranchConv
opt = parse_option()
logger_my = Logger(opt)
logprint = logger_my.log_printer
accprint = logger_my.log_printer.accprint
netprint = logger_my.log_printer.netprint
opt.save_folder = logger_my.weights_path
opt.print_interval = opt.print_freq
opt.logprint = logger_my.log_printer
timer = Timer(opt.epochs)
'''README-Huan:
This file is adapted from 'train_teacher.py'.
'''
# ---

def train(epoch, train_loader, model, criterion, optimizer, opt, trainable_list, print=print):
    """vanilla training"""
    trainable_list.train()
    num_step_per_epoch = len(train_loader)
    for idx, (input, target) in enumerate(train_loader):
        total_step = (epoch - 1) * num_step_per_epoch + idx

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        feat, logit = model(input, is_feat=True)
        cnt, total_loss = -1, 0
        acc1, loss = [], []
        for ix in opt.branch_layer: # example: [1,2,3]
            cnt += 1
            head = trainable_list[cnt]
            logit = head(feat[ix])
            loss_ = criterion(logit, target)
            total_loss += loss_
            acc1.append(cal_acc(logit, target).item())
            loss.append(loss_.item())
        if idx % opt.print_freq == 0:
            logtmp1 = ' '.join(['%.4f' % x for x in acc1])
            logtmp2 = ' '.join(['%.4f' % x for x in loss])
            print(f'Step {total_step} -- Train_accuracy {logtmp1} Train_loss {logtmp2}')
        # ===================backward=====================
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

def main():
    best_acc = 0
    best_acc_epoch = 0

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, _, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers,
            npy_set=opt.npy_set, augment=opt.augment, use_DA=opt.use_DA)
        n_cls = 100
        img_size = 32
    elif opt.dataset == 'svhn':
        train_loader, val_loader = get_svhn_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
        img_size = 32
    elif opt.dataset == 'tinyimagenet':
        train_loader, val_loader = get_tinyimagenet_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 200
        img_size = 64
    elif opt.dataset == 'imagenet':
        train_loader, val_loader = get_imagenet_dataloader(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=False)
        n_cls = 1000
        img_size = 224
    else:
        raise NotImplementedError(opt.dataset)

    # Set up model
    criterion = nn.CrossEntropyLoss()
    if opt.dataset == 'imagenet':
        model = eval('resnet_imagenet.%s' % opt.model)().cuda()
    else:
        model = model_dict[opt.model](num_classes=n_cls, img_size=img_size).cuda()
    
    # Load weights
    ckpt = torch.load(opt.pretrained)
    from collections import OrderedDict
    state_dict = OrderedDict()
    for k, v in ckpt['model'].items():
        if opt.transfer and 'fc.' in k or 'linear.' in k or 'classifier.' in k: # all names for FC layers
            continue
        if k.startswith('module.'):
            state_dict[k[7:]] = v
        else:
            state_dict[k] = v
    model.load_state_dict(state_dict, strict=False) # strict=False for transfer learning case
    logprint(f'Loading pretrained weights successfully: "{opt.pretrained}"')
    if opt.dataset != 'imagenet':
        acc1, loss = validate(val_loader, model, None, criterion, opt)
        logprint(f'Its accuracy: {acc1:.4f}')

    # fix the model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # add linear classifiers
    trainable_list = nn.ModuleList([])
    if opt.dataset in ['cifar100', 'svhn']:
        data = torch.randn(2, 3, 32, 32)
    if opt.dataset in ['tinyimagenet']:
        data = torch.randn(2, 3, 64, 64)
    if opt.dataset in ['imagenet']:
        data = torch.randn(24, 3, 224, 224)
    feat, _ = model(data.cuda(), is_feat=True)
    for ix in opt.branch_layer:
        logprint(f'Register branch: branch {ix} feature_shape: {feat[ix].shape}')
        classifier = BranchConv(feat[ix], n_class=n_cls, avgpool=False, n_fc=1)
        classifier = torch.nn.DataParallel(classifier)
        trainable_list.append(classifier)

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)



    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    logger = None # kd-Huan: we will use our own logger

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        logprint("==> training...")
        # --- kd-Huan:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        logprint("==> Set lr %s @ Epoch %d " % (lr, epoch))
        # ---

        time1 = time.time()
        train(epoch, train_loader, model, criterion, optimizer, opt, trainable_list, print=logprint)
        time2 = time.time()
        logprint('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        accprint('Predicted finish time: %s' % timer())

        # test branches
        acc1, loss = validate(val_loader, model, trainable_list, criterion, opt)
        logtmp1 = ' '.join(['%.4f' % x for x in acc1])
        logtmp2 = ' '.join(['%.4f' % x for x in loss])
        logprint(f'Epoch {epoch} -- Test_accuracy {logtmp1} Test_loss {logtmp2}')

        # save model
        state = {
            'epoch': epoch,
            'model': trainable_list,
            'state_dict': trainable_list.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_file = os.path.join(opt.save_folder, 'ckpt.pth')
        torch.save(state, save_file)

def validate(val_loader, model, trainable_list, criterion, opt):
    """validation"""
    top1, loss = [], [] # branch classifier acc and loss
    top1_model, loss_model = AverageMeter(), AverageMeter() # main network acc and loss

    # switch to evaluate mode
    model.eval()
    if trainable_list:
        trainable_list.eval()
        for _ in opt.branch_layer:
            top1 += [AverageMeter()]
            loss += [AverageMeter()]

    with torch.no_grad():
        for idx, (input, target) in enumerate(val_loader):
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                model = model.cuda()

            # compute output
            feat, output = model(input, is_feat=True)
            loss_ = criterion(output, target)
            top1_, _ = accuracy(output, target, topk=(1, 5))
            top1_model.update(top1_.item(), n=input.size(0))
            loss_model.update(loss_.item(), n=input.size(0))
            
            if trainable_list:
                cnt = -1
                for ix in opt.branch_layer: # example: [1,2,3]
                    cnt += 1
                    head = trainable_list[cnt]
                    output = head(feat[ix])
                    loss_ = criterion(output, target)
                    top1_, _ = accuracy(output, target, topk=(1, 5))
                    top1[cnt].update(top1_.item(), n=input.size(0))
                    loss[cnt].update(loss_.item(), n=input.size(0))
    
    if trainable_list:
        top1 = [x.avg for x in top1]
        loss = [x.avg for x in loss]
        return top1, loss
    else:
        return top1_model.avg, loss_model.avg

if __name__ == '__main__':
    main()