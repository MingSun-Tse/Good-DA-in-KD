"""
the general training framework
"""

from __future__ import print_function

import os
import configargparse
import socket
import time

# import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser
import models.resnet_imagenet as resnet_imagenet 

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.svhn import get_svhn_dataloaders, get_svhn_dataloaders_sample
from dataset.tinyimagenet import get_tinyimagenet_dataloaders, get_tinyimagenet_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader, get_dataloader_sample

from helper.util import adjust_learning_rate, get_teacher_name

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill, validate
from helper.pretrain import init
import numpy as np
from torchvision import models as tvmodels
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def load_teacher(model_path, n_cls):
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print(f'==> Load teacher weights successfully: "{model_path}"')
    return model

def load_auxiliary_classifiers(model_path, classifiers):
    ckpt = torch.load(model_path)
    if 'auxiliary_classifiers' in ckpt:
        classifiers.load_state_dict(ckpt['auxiliary_classifiers'])
        print(f'==> Load auxiliary classifiers weights successfully: "{model_path}"')
    else:
        print(f'==> Not found auxiliary classifiers weights in ckpt. Go on')
    return classifiers

# --- @mst
from option import opt
from smilelogging import Logger
logger = Logger(opt)
from utils import get_class_corr, Timer, smart_weights_load, get_n_params_, get_n_flops_
from branch import BranchConv
accprint = logger.accprint
netprint = logger.netprint
# ---

def main():
    best_acc = 0
    best_acc_epoch = 0
    epoch_start = 1
    opt.passer = {}

    # Set up model
    if opt.dataset == 'cifar100':
        n_cls, img_size = 100, 32
        model_t = load_teacher(opt.path_t, n_cls).cuda() # @mst: move this forward because model_t will be used later
        model_s = model_dict[opt.model_s](num_classes=n_cls, img_size=img_size).cuda()
    
    elif opt.dataset == 'svhn':
        n_cls, img_size = 10, 32
        model_t = load_teacher(opt.path_t, n_cls).cuda()
        model_s = model_dict[opt.model_s](num_classes=n_cls, img_size=img_size).cuda()
    
    elif opt.dataset == 'tinyimagenet':
        n_cls, img_size = 200, 64
        model_t = load_teacher(opt.path_t, n_cls).cuda()
        model_s = model_dict[opt.model_s](num_classes=n_cls, img_size=img_size).cuda()

    elif opt.dataset in ['imagenet', 'imagenet100']:
        img_size = 224
        n_cls = 1000 if opt.dataset == 'imagenet' else int(opt.dataset[8:])
        if opt.model_t_pretrained:
            model_t = eval('resnet_imagenet.%s' % opt.model_t)(num_classes=n_cls).cuda()
            smart_weights_load(model_t, opt.model_t_pretrained)
            print('==> Load pretrained teacher successfully: "%s"' % opt.model_t_pretrained)
        else:
            model_t = eval('resnet_imagenet.%s' % opt.model_t)(num_classes=n_cls).cuda()
            resnet34_torchvision_model = 'models/resnet34-333f7ec4.pth'
            if not os.path.exists(resnet34_torchvision_model):
                download = 'wget https://download.pytorch.org/models/resnet34-333f7ec4.pth -P models'
                os.system(download)
                print('==> Not found torchvision model. Download it at "%s"' % resnet34_torchvision_model)
            smart_weights_load(model_t, resnet34_torchvision_model)
            print('==> Load pretrained teacher successfully: Use offical torchvision model')
        
        model_s = model_dict[f'{opt.model_s}'](num_classes=n_cls, img_size=img_size).cuda()
    else:
        raise NotImplementedError
    opt.n_cls = n_cls # will be used later
    model_t = torch.nn.DataParallel(model_t)
    model_s = torch.nn.DataParallel(model_s)

    # Print number of parameters (weights and biases in conv and fc layers)
    n_params_t, n_flops_t = get_n_params_(model_t), get_n_flops_(model_t, img_size=img_size)
    n_params_s, n_flops_s = get_n_params_(model_s), get_n_flops_(model_s, img_size=img_size)
    print('n_params teacher: %.4f M  n_params student: %.4f M  compression: %.4f x (weights and biases  in conv and fc layers)' % (n_params_t/1e6, n_params_s/1e6, n_params_t/n_params_s))
    print('n_flops  teacher: %.4f G  n_flops  student: %.4f G  speedup    : %.4f x (weight MultiplyAdds in conv and fc layers)' % (n_flops_t/1e9, n_flops_s/1e9, n_flops_t/n_flops_s))

    # Resume or finetune student
    if opt.resume_student: 
        ckpt = torch.load(opt.resume_student)
        smart_weights_load(model_s, opt.resume_student)
        epoch_start = ckpt['epoch']
        print("==> Resume student successfully: '%s' @ Epoch %d Acc1: %.4f" % (opt.resume_student, epoch_start, ckpt['accuracy']))
    
    if opt.finetune_student: # deprecated! use 'opt.model_s_pretrained'
        ckpt = torch.load(opt.finetune_student)
        smart_weights_load(model_s, opt.finetune_student)
        if 'accuracy' in ckpt:
            acc = '%.4f' % ckpt['accuracy']
        elif 'best_acc' in ckpt:
            acc = '%.4f' % ckpt['best_acc'] # for backward compatibility
        else:
            acc = 'Unknown'
        print("==> Load pretrained student successfully: '%s' Acc1: %s" % (opt.finetune_student, acc))

    if opt.model_s_pretrained:
        smart_weights_load(model_s, opt.model_s_pretrained)
        print('==> Load pretrained student successfully: "%s"' % opt.model_s_pretrained)

    if opt.reinit_student:
        for name, module in model_s.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                module.reset_parameters()
        print(f'==> Reinit student randomly')
    
    if opt.fix_student:
        model_s.eval()
        for param in model_s.parameters():
            param.requires_grad = False
        print('==> Student freezed!')
    
    # Set up data loader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd', 'dcs+crd']:
            train_loader, val_loader, n_data, train_set = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode, 
                                                                               use_DA=opt.use_DA,
                                                                               opt=opt)
            train_loader2 = train_loader # temporary use to maintain interface
        else:
            train_loader, train_loader2, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True,
                                                                        use_DA=opt.use_DA,
                                                                        opt=opt)
    elif opt.dataset == 'svhn': # @mst
        if opt.distill in ['crd', 'dcs+crd']:
            train_loader, val_loader, n_data, train_set = get_svhn_dataloaders_sample(batch_size=opt.batch_size, 
                                                                                    num_workers=opt.num_workers,
                                                                                    k=opt.nce_k,
                                                                                    mode=opt.mode)
        else:
            train_loader, val_loader = get_svhn_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=False)
        train_loader2 = train_loader # temporary use to maintain interface

    elif opt.dataset == 'tinyimagenet':
        if opt.distill in ['crd', 'dcs+crd']:
            train_loader, val_loader, n_data, train_set = get_tinyimagenet_dataloaders_sample(batch_size=opt.batch_size, 
                                                                                    num_workers=opt.num_workers,
                                                                                    k=opt.nce_k,
                                                                                    mode=opt.mode)
        else:
            train_loader, val_loader = get_tinyimagenet_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=False)
        train_loader2 = train_loader # temporary use to maintain interface

    elif opt.dataset in ['imagenet', 'imagenet100']:
        if opt.distill in ['crd', 'dcs+crd']:
            train_loader, val_loader, n_data, _ = get_dataloader_sample(dataset=opt.dataset,
                                                                        batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        k=opt.nce_k,
                                                                        is_sample=True)
        else:
            train_loader, val_loader, n_data = get_imagenet_dataloader(dataset=opt.dataset,
                                                                        batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        train_loader2 = train_loader # temporary use
    else:
        raise NotImplementedError(opt.dataset)

    data = torch.randn(24, 3, img_size, img_size).cuda()
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s) # @mst: this module_list records all modules, which will be indexed later
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    # @mst
    if hasattr(opt, 'utils') and opt.utils.check_ce_var:
        opt.ce = 0 # cross-entropy for the current sampled training subset
        opt.all_avg_ce = [] # a list of all average cross-entropy loss values
        
        opt.n_samples = 0
        opt.all_risk = []
        
        opt.prob = []
        opt.all_avg_prob = []
        opt.entropy = []

    # @mst
    if hasattr(opt, 'utils') and opt.utils.check_ce_var_v2:
        opt.cov1_history = []
        opt.cov2_history = []
        opt.corr1_history = []
        opt.corr2_history = []

    # @mst
    if hasattr(opt, 'utils') and opt.utils.check_ce_var_v3:
        opt.one_RV_sample = []
        opt.all_RV_samples = []
        opt.one_RV_sample_approx = []
        opt.all_RV_samples_approx = []

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T, opt.kd_S)
    
    # Only testing
    if opt.only_test:
        if 'teacher' in opt.only_test:
            teacher_acc1, teacher_acc5, teacher_testloss = validate(val_loader, model_t, criterion_cls, opt)
            print(f'Test teacher. Acc1 {teacher_acc1} Acc5 {teacher_acc5} TestLoss {teacher_testloss:.6f}')
        if 'student' in opt.only_test:
            student_acc1, student_acc5, student_testloss = validate(val_loader, model_s, criterion_cls, opt)
            print(f'Test student. Acc1 {student_acc1} Acc5 {student_acc5} TestLoss {student_testloss:.6f}')
        exit(0)
        
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T, opt.kd_S)
    
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)

    # --- @mst: merge 'crd' with 'dcs'
    elif opt.distill in ['dcs', 'crd', 'dcs+crd']:
        # dcs part
        if opt.distill in ['dcs', 'dcs+crd']:
            auxiliary_classifiers_t = nn.ModuleList([]) # for saving checkpoint
            auxiliary_classifiers_s = nn.ModuleList([]) # for saving checkpoint
            criterion_kd = DistillKL(opt.kd_T, opt.kd_S)
            for ixt, ixs in zip(opt.branch_layer_T, opt.branch_layer_S):
                f_t = feat_t[ixt]
                f_s = feat_s[ixs]
                classifier_t = BranchConv(f_t, n_class=n_cls, n_fc=opt.n_branch_fc_T, width=opt.branch_width_T)
                classifier_s = BranchConv(f_s, n_class=n_cls, n_fc=opt.n_branch_fc_S, width=opt.branch_width_S)
                
                # Orth init
                if opt.head_init == 'orth':
                    for net_ in [classifier_t, classifier_s]:
                        for module in net_.modules():
                            if isinstance(module, nn.Linear):
                                nn.init.orthogonal_(module.weight)
                    print(f'Use orth init for head FC')
                    
                classifier_t = torch.nn.DataParallel(classifier_t)
                classifier_s = torch.nn.DataParallel(classifier_s)
                auxiliary_classifiers_t.append(classifier_t)
                auxiliary_classifiers_s.append(classifier_s)
            
            trainable_list.append(auxiliary_classifiers_s)
            if not opt.fix_T_heads:
                trainable_list.append(auxiliary_classifiers_t)

            auxiliary_classifiers_t = load_auxiliary_classifiers(opt.path_t, auxiliary_classifiers_t)
            opt.passer['auxiliary_classifiers_t'] = auxiliary_classifiers_t
            opt.passer['auxiliary_classifiers_s'] = auxiliary_classifiers_s

            # get params
            params = 0
            for ac in auxiliary_classifiers_t:
                params += get_n_params_(ac)
            print(f'Teacher auxiliary classifier head params: {params/10**6} M')
        
        # crd part
        if opt.distill in ['crd', 'dcs+crd']:
            if opt.crd_multiheads: # @mst
                criterion_kd = nn.ModuleList([])
                for ixt, ixs in zip(opt.branch_layer_T, opt.branch_layer_S):
                    f_t = feat_t[ixt]
                    f_s = feat_s[ixs]
                    opt.s_dim = f_s.size(1) # NCHW
                    opt.t_dim = f_t.size(1) # NCHW
                    opt.n_data = n_data
                    criterion_ = CRDLoss(opt)
                    criterion_kd.append(criterion_)
                    module_list.append(criterion_.embed_s)
                    module_list.append(criterion_.embed_t)
                    trainable_list.append(criterion_.embed_s)
                    trainable_list.append(criterion_.embed_t)
            else:
                opt.s_dim = feat_s[-1].shape[1]
                opt.t_dim = feat_t[-1].shape[1]
                opt.n_data = n_data
                criterion_kd = CRDLoss(opt)
                module_list.append(criterion_kd.embed_s)
                module_list.append(criterion_kd.embed_t)
                trainable_list.append(criterion_kd.embed_s)
                trainable_list.append(criterion_kd.embed_t)
            
            # @mst: use pretrained embed network
            if opt.pretrained_embed:
                ckpt_embed = torch.load(opt.pretrained_embed)
                module_list[1].load_state_dict(ckpt_embed['embed_s'])
                module_list[2].load_state_dict(ckpt_embed['embed_t'])
                print("==> Load pretrained embed network successfully: '%s'" % opt.pretrained_embed)
            
            # freeze embed network (usually when pretrained embed network is provided)
            if opt.fix_embed:
                module_list[1].eval()
                module_list[2].eval()
                for param in module_list[1].parameters():
                    param.requires_grad = False
                for param in module_list[2].parameters():
                    param.requires_grad = False
                print("==> Freeze embed network")
    # ---

    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay,
                          nesterov=False)
    
    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    # @mst: print models for check
    if not opt.debug:
        netprint(module_list)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    if opt.test_teacher:
        print('==> testing teacher ...')
        teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
        print('teacher accuracy: %.2f' % teacher_acc.item())

    # routine
    timer = Timer(opt.epochs)
    for epoch in range(epoch_start, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")
        
        # @mst:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("==> Set lr %s @ Epoch %d " % (lr, epoch))

        # @mst: adapative weight decay
        if opt.weight_decay_schedule:
            from utils import strdict_to_dict
            sch = strdict_to_dict(opt.weight_decay_schedule, ttype=float) # ['0':0.0005, '150':0.0001]
            wdepochs = sorted([int(x) for x in sch.keys()]) # example: [0, 30, 45]
            wd = sch[str(wdepochs[-1])]
            for i in range(len(wdepochs) - 1):
                if wdepochs[i] < epoch <= wdepochs[i+1]:
                    wd = sch[str(wdepochs[i])]
                    break
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = wd
            print(f'==> Set wd {wd} @ Epoch {epoch}')

        time1 = time.time()
        if opt.test_loader_in_train:
            train_loader = val_loader
            print(f'==> Note: Using test loader during training!')
            time.sleep(2)
        opt.passer['logger'] = logger
        train_acc, train_loss = train_distill(epoch, train_loader, train_loader2, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        # logger.log_value('test_acc', test_acc, epoch)
        # logger.log_value('test_loss', test_loss, epoch)
        # logger.log_value('test_acc_top5', test_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_epoch = epoch
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': best_acc, 
            }
            if opt.distill == 'crd': # @mst: save the embedding network
                state['embed_s'] = module_list[1].state_dict()
                state['embed_t'] = module_list[2].state_dict()
           
            if 'dcs' in opt.distill:
                state['auxiliary_classifiers'] = auxiliary_classifiers_s.state_dict()
            
            save_file = os.path.join(logger.weights_path, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # --- @mst:
        accprint("Acc1 %.4f Acc5 %.4f TestLoss %.6f Epoch %d (after update) lr %s (Best_Acc1 %.4f @ Epoch %d)" % 
            (test_acc, test_acc_top5, test_loss, epoch, lr, best_acc, best_acc_epoch))
        print('predicted finish time: %s' % timer())
        # ---

        # Regular saving
        print('==> Saving...')
        state = {
            'epoch': epoch,
            'model': model_s.state_dict(),
            'accuracy': test_acc,
            'optimizer': optimizer.state_dict(),
            'ExpID': logger.ExpID,
        }
        if opt.distill == 'crd': # @mst: save the embedding network
            state['embed_s'] = module_list[1].state_dict()
            state['embed_t'] = module_list[2].state_dict()
        
        if 'dcs' in opt.distill:
            state['auxiliary_classifiers'] = auxiliary_classifiers_s.state_dict()
        
        save_file = os.path.join(logger.weights_path, 'ckpt.pth')
        torch.save(state, save_file)
        
        if opt.save_freq > 0 and epoch % opt.save_freq == 0:
            save_file = os.path.join(logger.weights_path, f'ckpt_{epoch}.pth')
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy: %.4f' % best_acc.item())

    # save model
    # state = {
    #     'opt': opt,
    #     'model': model_s.state_dict(),
    # }
    # save_file = os.path.join(logger.weights_path, '{}_last.pth'.format(opt.model_s))
    # torch.save(state, save_file)
    
if __name__ == '__main__':
    # Scp results
    scp_script = 'scripts/scp_experiments_to_hub.sh'
    if not opt.debug and os.path.exists(scp_script):
        from smilelogging.utils import scp_experiment
        scp_experiment(scp_script, logger, opt, mv=False)
        print('==> Initial scp done')
    main()
    # Scp results
    if not opt.debug and os.path.exists(scp_script):
        from smilelogging.utils import scp_experiment
        scp_experiment(scp_script, logger, opt, mv=True)
        print('==> Final scp done')