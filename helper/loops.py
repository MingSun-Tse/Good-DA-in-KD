from __future__ import print_function, division

import sys, time, os, math, numpy as np, copy
import functools

import torch
torch.multiprocessing.set_sharing_strategy('file_system') # https://github.com/pytorch/pytorch/issues/973
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn as nn
import pdb

from .util import AverageMeter, accuracy, CeilingRatioScheduler
from utils import cal_correlation, cal_acc, strdict_to_dict, denormalize_image, make_one_hot, LossLine

tensor2list = lambda x: x.data.cpu().numpy().tolist()
tensor2np = lambda x: x.data.cpu().numpy()

def get_grad_scaler():
    torch_versions_amp = ['1.9.', '1.10'] # @mst-TODO: ugly impl. May be improved later
    if torch.__version__[:4] in torch_versions_amp:
        scaler = torch.cuda.amp.GradScaler()
        return scaler
    else:
        print(f'Error: Pytorch version is too old for amp, exit. Desired versions are {torch_versions_amp}')
        exit(1)

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        # refers to the official pytorch usage of AutoAugment: 
        # https://pytorch.org/vision/master/transforms.html#torchvision.transforms.AutoAugment
        if opt.mix_mode in ['autoaugment']:
            from torchvision.transforms.autoaugment import AutoAugment
            from torchvision.transforms.autoaugment import AutoAugmentPolicy
            from utils import denormalize_image, normalize_image
            if opt.dataset in ['imagenet', 'imagenet100']:
                policy = AutoAugmentPolicy.IMAGENET
                raise NotImplementedError
            elif opt.dataset in ['cifar100', 'cifar10']:
                policy = AutoAugmentPolicy.CIFAR10
                from dataset.cifar100 import MEAN, STD
            autoaug = AutoAugment(policy=policy)
            # ******************************************
            input = denormalize_image(input, mean=MEAN, std=STD)
            input = (input.mul(255)).type(torch.uint8) # autoaugment in pytorch demands the input is int8 if the input is a tensor
            input = autoaug(input).type(torch.float32).div(255)
            input = normalize_image(input, mean=MEAN, std=STD)
            # ******************************************
        
        
        if opt.train_auxiliary_classifiers:
            model.eval()
            with torch.no_grad():
                feat, _ = model(input, is_feat=True, preact=False)
            auxiliary_classifiers = opt.passer['auxiliary_classifiers']
            loss, acc_all_heads = 0, []
            for ix, f_ix in enumerate(opt.branch_layer_T):
                head = auxiliary_classifiers[ix]
                head_logit = head(feat[f_ix])
                loss += criterion(head_logit, target)

                # Print acc for check
                acc = accuracy(head_logit, target)[0]
                acc_all_heads += [acc]
            output = head_logit # take the last head logit as output to calculate train acc
        else:
            output = model(input)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # Print
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            
            if opt.train_auxiliary_classifiers:
                print(f'Auxiliary classifier acc: ' + ' '.join(['%.4f' % x for x in acc_all_heads]))
            sys.stdout.flush()

    return top1.avg, losses.avg

def train_distill(epoch, train_loader, train_loader2, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    logger = opt.passer['logger']

    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()
    # @mst
    if opt.fix_student:
        module_list[0].eval()
    if opt.fix_embed:
        module_list[1].eval()
        module_list[2].eval()
    # ---

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t  = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    
    if opt.two_loader:
        train_loader2_list = list(enumerate(train_loader2))

    # @mst: mixed precision training
    if opt.amp:
        grad_scaler = get_grad_scaler()

    # @mst: use ceiling_ratio_schedule
    opt.ceiling_ratio, opt.floor_ratio = -1, -1
    if opt.ceiling_ratio_schedule:
        ceiling_ratio_scheduler = CeilingRatioScheduler(strdict_to_dict(opt.ceiling_ratio_schedule, ttype=float))
        floor_ratio_scheduler = CeilingRatioScheduler(strdict_to_dict(opt.floor_ratio_schedule, ttype=float))
        opt.ceiling_ratio = ceiling_ratio_scheduler.get_ceiling_ratio(epoch)
        opt.floor_ratio = floor_ratio_scheduler.get_ceiling_ratio(epoch)
        print('==> Use ceiling_ratio %s floor_ratio %s Epoch %d (ceiling ratio of the remaining part in CutMix)' % (opt.ceiling_ratio, opt.floor_ratio, epoch))

    # @mst: use my sampling
    n_step_per_epoch = len(train_loader)

    # @mst: use smilelogging loss line for logging
    from smilelogging.utils import LossLine
    lossline = LossLine()
    for idx, data in enumerate(train_loader):
        opt.epoch, opt.step = epoch, idx # for later use
        opt.total_step = idx + n_step_per_epoch * (epoch - 1) # total step. note, epoch starts from 1

        if opt.distill in ['crd', 'dcs+crd']:
            input, target, index, contrast_idx = data # index: the index in the dataset for the current batch. contrast_idx: [batch_size, 1 + n_negatives]
        else:
            input, target = data[:2]
            index = data[2] if len(data) == 3 else None
        
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()
            if opt.distill in ['crd', 'dcs+crd']:
                contrast_idx = contrast_idx.cuda()
            if index is not None:
                index = index.cuda()

        # @mst: add online DA through editing the input images
        if opt.lr_DA > 0:
            assert opt.distill == 'kd'
            if opt.two_loader:
                _, data2 = train_loader2_list[idx]
                input2, target2, *_ = data2
                input2, target2 = input2.cuda(), target2.cuda()
            else:
                input2, target2 = input, target
                
            input2, loss_DA = MyDA2(input2, model_t, model_s, opt.lr_DA) # data augmentation
            if loss_DA:
                lossline.update("loss_DA", loss_DA.item(), '.4f')
            
            if opt.online_augment:
                input  = torch.cat([input,   input2], dim=0)
                target = torch.cat([target, target2], dim=0)
            else:
                input  = input2
                target = target2

        # Modify input for student: teacher and student will have different inputs
        input_s = input
        if opt.modify_student_input in ['cutout', 'masking']:
            func = eval(f'{opt.modify_student_input}')
            input_s, *_ = func(input, opt=opt)
            if opt.stack_input: # 'input_s' as an extra input
                input_s = torch.cat([input, input_s], dim=0)
                input   = torch.cat([input, input  ], dim=0) # since student's input is changed, so be the teacher's input and the target
                target  = torch.cat([target, target], dim=0)

        # ===================forward=====================
        # with torch.cuda.amp.autocast():
        preact = False
        if opt.distill in ['abound']:
            preact = True
        
        # Get internal features and logits
        if opt.distill == 'kd':
            logit_s = model_s(input_s)
            with torch.no_grad():
                logit_t = model_t(input)
        else:
            feat_s, logit_s = model_s(input_s, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]
        
        # @mst: Advanced data augmentation techniques
        if opt.lw_mix is not None and epoch <= opt.input_mix_no_kld_epoch:
            input_mix, target_mix, logit_t_mix, logit_s_mix, rand_index, lam = \
                get_mixed_input(opt.mix_mode, input, target, logit_t, logit_s.detach(), model_t, model_s, opt)
            
            # sanity-check mix ratio
            if idx == 0: 
                print(lam)
            
            # save entropy log for later analysis
            if opt.save_entropy_log_step > 0 and opt.total_step % opt.save_entropy_log_step == 0:
                save_path = '%s/entropy_log.npy' % logger.log_path
                np.save(save_path, np.asarray(opt.entropy_log))
                print('==> entropy log saved: "%s"' % save_path)

            if idx > opt.save_img_interval and idx % opt.save_img_interval == 0:
                save_path = f'{logger.gen_img_path}/epoch{epoch}_step{idx}_input_mix.png'
                vutils.save_image(input_mix[:8], save_path, nrow=4, padding=0, normalize=True, scale_each=True)

        total_loss = torch.Tensor([0]).cuda() # total loss


        # [Loss #1] cls loss
        bs_CE = int(target.size(0) * opt.ratio_CE_loss) # @mst: bs_CE: batch_size that applies CE loss
        # This is useful when we want to apply CE loss to only a part of the batch data
        loss_cls = torch.Tensor([0]).cuda() if bs_CE == 0 else criterion_cls(logit_s[:bs_CE], target[:bs_CE])
        if opt.gamma:
            total_loss += opt.gamma * loss_cls
            lossline.update(f'loss_cls(*{opt.gamma})', loss_cls.item(), '.6f')

        # [Loss #2] KL div loss
        if opt.lw_mix is None:
            loss_div = criterion_div(logit_s, logit_t)
        
        else: # use data-mixing
            loss_div = torch.Tensor([0]).cuda()
            lw1, lw2, lw3 = opt.lw_mix
            if lw1:
                # kld loss for original input
                loss_ = criterion_div(logit_s, logit_t.detach())
                loss_div += loss_ * lw1 
                lossline.update(f'loss_div-oriKLD(*{lw1})', loss_.item(), '.6f')
            
            if lw2:
                # CE loss for mixed input
                if opt.mix_mode == 'cutmix':
                    input_mix_logit_s = model_s(input_mix)
                    loss_ = criterion_cls(input_mix_logit_s, target) * lam + criterion_cls(input_mix_logit_s, target[rand_index]) * (1. - lam)
                    # refer to: https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L240
                else:
                    raise NotImplementedError
                    # bce_loss = nn.BCELoss() # refer to: https://github.com/vikasverma1077/manifold_mixup/blob/master/supervised/main.py
                    # loss_ = bce_loss(F.softmax(logit_s, dim=1), target_mix.cuda())
                loss_div += loss_ * lw2
                lossline.update(f'loss_div-mixCE(*{lw2})', loss_.item(), '.6f')
            
            if lw3 and epoch <= opt.input_mix_no_kld_epoch:
                # kld loss for mixed input
                if opt.t_output_as_target_for_input_mix:
                    with torch.no_grad():
                        input_mix_logit_t = model_t(input_mix)
                    input_mix_target = input_mix_logit_t
                else:
                    input_mix_target = logit_t_mix
                input_mix_logit_s = model_s(input_mix)
                loss_ = criterion_div(input_mix_logit_s, input_mix_target.detach())
                loss_div += loss_ * lw3
                lossline.update(f'loss_div-mixKLD(*{lw3})', loss_.item(), '.6f')

        # Check teacher's avg probability
        if hasattr(opt, 'utils') and opt.utils.check_ce_var:
            p = F.softmax(logit_t, dim=-1)
            opt.prob += [p] # [batch_size, num_classes]
            opt.entropy += [(-p * torch.log(p)).sum(dim=-1)] # [batch_size]
            if opt.lw_mix is not None:
                p = F.softmax(input_mix_logit_t, dim=-1)
                opt.prob += [p] # [batch_size, num_classes]
                opt.entropy += [(-p * torch.log(p)).sum(dim=-1)] # [batch_size]
            num = 5 if opt.lw_mix is None else 10 # 10 is empirically set
            if len(opt.prob) >= num: 
                prob = torch.cat(opt.prob, dim=0) # [..., num_classes]
                entropy = torch.cat(opt.entropy, dim=0)
                avg_prob = prob.mean(dim=0) # [num_classes]
                opt.all_avg_prob += [avg_prob]
                opt.prob = [] # reset 

                all_avg_prob = torch.stack(opt.all_avg_prob, dim=0) # [Num, num_classes]
                avg_prob_std = all_avg_prob.std(dim=0)
                # std_str = ' '.join(['%.6f' % x for x in tensor2list(avg_prob_std)])
                std_str = '%.6f' % avg_prob_std.mean().item()
                print(f'Check T prob: NumOfSampledStd {len(opt.all_avg_prob)} Epoch {epoch} Step {idx} TotalStep {opt.total_step} MeanStd {std_str} MeanEntropy {entropy.mean().item():.6f}')
        
        # Here we implement the idea of directly calculating the covariance of teacher's outputs
        if hasattr(opt, 'utils') and opt.utils.check_ce_var_v2:
            logit_total = torch.cat([input_mix_logit_t, logit_t], dim=0) # [2*batch_size, num_classes]
            p = F.softmax(logit_total, dim=-1).data.cpu().numpy() # [2*batch_size, num_classes]
            # Covariance
            cov1 = np.cov(p, rowvar=True) # [batch_size, batch_size]
            cov2 = np.cov(p, rowvar=False) # [num_classes, num_classes]
            cov1_mean, cov1_abs_mean = cov1.mean(), np.abs(cov1).mean()
            cov2_mean, cov2_abs_mean = cov2.mean(), np.abs(cov2).mean()

            # Correlation
            corr1 = np.corrcoef(p, rowvar=True) # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
            corr2 = np.corrcoef(p, rowvar=False)
            corr1_mean, corr1_abs_mean = corr1.mean(), np.abs(corr1).mean()
            corr2_mean, corr2_abs_mean = corr2.mean(), np.abs(corr2).mean()
            
            # Collect all history to do averaging later
            opt.cov1_history += [[cov1_mean, cov1_abs_mean]]
            opt.cov2_history += [[cov2_mean, cov2_abs_mean]]
            opt.corr1_history += [[corr1_mean, corr1_abs_mean]]
            opt.corr2_history += [[corr2_mean, corr2_abs_mean]]
            
            # Print
            if idx % 10 == 0:
                cov1_mean_ = np.array(opt.cov1_history)[:, 0].mean()
                cov2_mean_ = np.array(opt.cov2_history)[:, 0].mean()
                cov1_abs_mean_ = np.array(opt.cov1_history)[:, 1].mean()
                cov2_abs_mean_ = np.array(opt.cov2_history)[:, 1].mean()

                corr1_mean_ = np.array(opt.corr1_history)[:, 0].mean()
                corr2_mean_ = np.array(opt.corr2_history)[:, 0].mean()
                corr1_abs_mean_ = np.array(opt.corr1_history)[:, 1].mean()
                corr2_abs_mean_ = np.array(opt.corr2_history)[:, 1].mean()

                print(f'Check T prob V2: Epoch {epoch} Step {idx} TotalStep {opt.total_step} Cov1_mean {cov1_mean_:.8f} \
Cov1_abs_mean {cov1_abs_mean_:.8f} Cov2_mean {cov2_mean_:.8f} Cov2_abs_mean {cov2_abs_mean_:.8f}')
                print(f'Check T prob V2: Epoch {epoch} Step {idx} TotalStep {opt.total_step} Corr1_mean {corr1_mean_:.8f} \
Corr1_abs_mean {corr1_abs_mean_:.8f} Corr2_mean {corr2_mean_:.8f} Corr2_abs_mean {corr2_abs_mean_:.8f}')

        if hasattr(opt, 'utils') and opt.utils.check_ce_var_v3:
            r"""For NIPS'22 rebuttal. We consider the student model to calculate the metric.
            """
            p_t = F.softmax(logit_t, dim=-1) # [batch_size]
            logp_s = F.log_softmax(logit_s, dim=-1) # [batch_size]
            logp_t = F.log_softmax(logit_t, dim=-1) # [batch_size]
            q = -(p_t * logp_s).sum(dim=-1) # [batch_size]
            q_approx = -(p_t * logp_t).sum(dim=-1) # [batch_size]
            opt.one_RV_sample += q.cpu().data.numpy().tolist() # Original input, [batch_size]
            opt.one_RV_sample_approx += q_approx.cpu().data.numpy().tolist() # Original input, [batch_size]

            if opt.lw_mix is not None:
                p_t = F.softmax(input_mix_logit_t, dim=-1)
                logp_s = F.log_softmax(input_mix_logit_s, dim=-1)
                logp_t = F.log_softmax(input_mix_logit_t, dim=-1)
                q = -(p_t * logp_s).sum(dim=-1) # [batch_size]
                q_approx = -(p_t * logp_t).sum(dim=-1) # [batch_size]
                opt.one_RV_sample += q.cpu().data.numpy().tolist() # Augmented input, [batch_size]. Now, shape [2*batch_size]
                opt.one_RV_sample_approx += q_approx.cpu().data.numpy().tolist() # Augmented input, [batch_size]. Now, shape [2*batch_size]
            
            if len(opt.one_RV_sample) >= 6400: # This 51200 is empirically set
                opt.all_RV_samples += [opt.one_RV_sample]
                opt.all_RV_samples_approx += [opt.one_RV_sample_approx]
                opt.one_RV_sample = [] # Reset
                opt.one_RV_sample_approx = [] # Reset

                if len(opt.all_RV_samples) >= 100: # 10 is empirically set
                    all_RV_samples = np.array(opt.all_RV_samples)
                    all_RV_samples = all_RV_samples / all_RV_samples.max(axis=0) # Normalization!!
                    num_RVs = all_RV_samples.shape[1]
                    cov = np.cov(all_RV_samples, rowvar=False)

                    all_RV_samples_approx = np.array(opt.all_RV_samples_approx)
                    all_RV_samples_approx = all_RV_samples_approx / all_RV_samples_approx.max(axis=0) # Normalization!!
                    cov_approx = np.cov(all_RV_samples_approx, rowvar=False)

                    cov_sum, cov_approx_sum = 0, 0
                    for i in range(num_RVs):
                        for j in range(i):
                            cov_sum += cov[i, j]
                            cov_approx_sum += cov_approx[i, j]
                    print(f'cov shape: {cov.shape}')
                    print(f'Check Covariance: NumOfSamples {len(opt.all_RV_samples)} Epoch {epoch} Step {idx} \
TotalStep {opt.total_step} Cov {cov_sum / num_RVs / num_RVs :.10f} Cov_approx {cov_approx_sum / num_RVs / num_RVs :.10f}')
                    exit()
        
        if opt.alpha:
            total_loss += opt.alpha * loss_div
            lossline.update(f'loss_div(*{opt.alpha})', loss_div.item(), '.6f')
        
        # [Loss #3] other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = torch.Tensor([0]).cuda()
        
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        
        # @mst: merge 'crd' with 'dcs'
        elif opt.distill in ['dcs', 'crd', 'dcs+crd']:
            auxiliary_classifiers_t = opt.passer['auxiliary_classifiers_t']
            auxiliary_classifiers_s = opt.passer['auxiliary_classifiers_s']
            if opt.fix_T_heads:
                auxiliary_classifiers_t.eval()

            # dcs part
            if opt.distill in ['dcs', 'dcs+crd']:
                loss_kd = torch.Tensor([0]).cuda()
                if epoch <= opt.epoch_stop_head_kd_loss:
                    loss_kd_pool, cnt, loss_branch_t = [], -1, 0
                    acc_all_heads_T, acc_all_heads_S = [], []
                    for ixt, ixs, drop_rate in zip(opt.branch_layer_T, opt.branch_layer_S, opt.branch_dropout_rate):
                        cnt += 1
                        head_s = auxiliary_classifiers_s[cnt] # the first is student model
                        l_s = head_s(feat_s[ixs]) # logit of student head

                        # print acc, student branch
                        accS = accuracy(l_s, target)[0]
                        acc_all_heads_S.append(accS)

                        # teacher classifier
                        if opt.s_branch_target == 't_branch':
                            head_t = auxiliary_classifiers_t[cnt]
                            l_t = head_t(feat_t[ixt]) # logit of teacher head
                            loss_branch_t += criterion_cls(l_t, target)
                            accT = accuracy(l_t, target)[0]
                            acc_all_heads_T.append(accT)

                        # student classifier
                        if opt.s_branch_target == 't_branch':
                            loss_tmp = criterion_div(l_s, l_t.detach()) * opt.lw_branch_kld + criterion_cls(l_s, target) * opt.lw_branch_ce
                        elif opt.s_branch_target == 't_logits':
                            loss_tmp = criterion_div(l_s, logit_t.detach()) * opt.lw_branch_kld + criterion_cls(l_s, target) * opt.lw_branch_ce
                        else:
                            raise NotImplementedError
                        if cnt == 0: # first stage
                            loss_tmp_1st = loss_tmp

                        # branch kd loss dropout
                        if np.random.rand() >= drop_rate: # rand: [0, 1)
                            loss_kd_pool.append(loss_tmp)

                    # add loss_branch_t to the total loss to train the branches of teacher
                    if not opt.fix_T_heads:
                        total_loss += loss_branch_t
                    
                    # if all internal kd losses are dropped, at least use the 1st stage kd loss
                    loss_kd = loss_tmp_1st if len(loss_kd_pool) == 0 else sum(loss_kd_pool)
                loss_dcs = loss_kd # backup, for use in 'dcs+crd'
            
            # crd part
            if opt.distill in ['crd', 'dcs+crd']:
                # @mst: apply CRD to multi-heads
                if opt.crd_multiheads:
                    loss_kd, cnt = torch.Tensor([0]).cuda(), -1
                    for ixt, ixs in zip(opt.branch_layer_T, opt.branch_layer_S):
                        cnt += 1
                        f_t = feat_t[ixt]
                        f_s = feat_s[ixs]
                        f_t = F.avg_pool2d(f_t, kernel_size=f_t.size(2))
                        f_s = F.avg_pool2d(f_s, kernel_size=f_s.size(2))
                        loss_kd += criterion_kd[cnt](f_s, f_t, index, contrast_idx)
                else:
                    f_s, f_t = feat_s[-1], feat_t[-1]
                    loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
                    loss_crd = loss_kd # backup, for use in 'dcs+crd'

            # # -- @mst: dcs+crd, not finished!
            # # Modify student input and add it as extra input. Here is the part to address the extra input
            # if opt.modify_student_input and opt.stack_input:
            #     assert opt.distill in ['dcs', 'dcs+crd'] # must have dcs
            #     func = eval(f'{opt.modify_student_input}')
            #     new_input, *_ = func(input, opt=opt)
            #     new_feat_s, _ = model_s(new_input)
            #     with torch.no_grad():
            #         new_feat_t, _ = model_t(new_input)
            #     if opt.stack_input: # 'input_s' as an extra input
            #         input_s = torch.cat([input, input_s], dim=0)
            #         input   = torch.cat([input, input  ], dim=0) # since student's input is changed, so be the teacher's input and the target
            #         target  = torch.cat([target, target], dim=0)
            # # --

        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = torch.Tensor([0]).cuda()
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = torch.Tensor([0]).cuda()
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)
        
        # Loss weight for kd loss beyond KL div
        if opt.beta:
            if opt.distill == 'dcs+crd':
                total_loss += opt.beta * loss_crd + opt.lw_dcs * loss_dcs
                lossline.update('loss_CRD(*%s)' % opt.beta, loss_crd.item(), '.4f')
                lossline.update('loss_DCS(*%s)' % opt.lw_dcs, loss_dcs.item(), '.4f')
            else:
                total_loss += opt.beta * loss_kd
                lossline.update('loss_OtherKD(*%s)' % opt.beta, loss_kd.item(), '.4f')

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(total_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        if opt.learning_rate > 0: # if lr is 0, no need to update
            optimizer.zero_grad()

            if opt.amp:
                grad_scaler.scale(total_loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print(f'Epoch: [{epoch:d}][{idx:d}/{len(train_loader):d}] '
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f}) | ' + lossline.format()
            )
            if opt.distill in ['dcs', 'dcs+crd']:
                print(f'    T Auxiliary classifier acc: ' + ' '.join(['%.4f' % x for x in acc_all_heads_T]))
                print(f'    S Auxiliary classifier acc: ' + ' '.join(['%.4f' % x for x in acc_all_heads_S]))
            sys.stdout.flush()

    return top1.avg, losses.avg

def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            if 'auxiliary_classifiers' in opt.passer:
                auxiliary_classifiers = opt.passer['auxiliary_classifiers']
                auxiliary_classifiers.eval()
                feat, _ = model(input, is_feat=True, preact=False)
                f = feat[opt.branch_layer_T[-1]]
                output = auxiliary_classifiers[-1](f)
            else:
                output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if idx % opt.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            #            idx, len(val_loader), batch_time=batch_time, loss=losses,
            #            top1=top1, top5=top5))

        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

# @mst: refer to cutmix official impel: 
# https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L279
def rand_bbox(size, lam):
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
    
def simpler_rand_bbox(size, lam=None):
    H, W = size[2], size[3]
    bbx1 = np.random.randint(0, W) # [0, W-1]
    bby1 = np.random.randint(0, H) # [0, H-1]
    bbx2 = np.random.randint(bbx1+1, W+1) # [bbx1+1, W], bbx2 must be larger than bbx1 by at least 1, otherwise the bbox will be none.
    bby2 = np.random.randint(bby1+1, H+1)
    return bbx1, bby1, bbx2, bby2

def square_rand_bbox(size, lam): # lam: area ratio of the remaining part
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    bbx1 = np.random.randint(0, W - cut_w + 1)
    bby1 = np.random.randint(0, H - cut_h + 1)
    return bbx1, bby1, bbx1+cut_w, bby1+cut_h

def mixup(input, target, logit_t, logit_s, model_t, model_s, opt):
    r"""Refer to official mixup impl.: https://github.com/facebookresearch/mixup-cifar10
    """
    alpha = 1.0 # according to official mixup impl., alpha is 1 in default 
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    rand_index = torch.randperm(input.size()[0]).cuda()
    input_mix = lam * input + (1 - lam) * input[rand_index, :]
    target_mix, logit_t_mix, logit_s_mix = None, None, None # placeholder to maintain interface
    return input_mix, target_mix, logit_t_mix, logit_s_mix, rand_index, lam


def cutmix(input, target, logit_t, logit_s, model_t, model_s, opt):
    '''Refer to official cutmix impl.: https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L234
    '''
    batch_size, _, h, w = input.size()
    
    # get random bounding box
    beta = 1.0
    lam = np.random.beta(beta, beta) # when beta=1, it degrades to the uniform dist
    if opt.ceiling_ratio != -1:
        # lam = (2 * ceiling_ratio - 1) * lam + (1 - ceiling_ratio) # (0, 1) -> (1-ceiling_ratio, ceiling_ratio)
        # lam = lam * ceiling_ratio                                 # (0, 1) -> (1-ceiling_ratio, ceiling_ratio)
        lam = opt.floor_ratio + (opt.ceiling_ratio - opt.floor_ratio) * lam     # (0, 1) -> (floor_ratio, ceiling_ratio)
    bbox_func = eval(opt.bbox)
    bbx1, bby1, bbx2, bby2 = bbox_func(input.size(), lam) # lam is the area ratio of the remaining part over the original image

    # get shuffle index
    rand_index = torch.randperm(batch_size).cuda()
    while 0 in (rand_index - torch.arange(batch_size).cuda()): # a small change from original cutmix impl.: use non-overlap rand_index
        rand_index = torch.randperm(input.size()[0]).cuda()

    # get new image
    mask = torch.ones((h, w)).cuda()
    mask[bby1: bby2, bbx1: bbx2] = 0
    input_mix = input * mask + input[rand_index] * (1 - mask)

    # adjust lambda to exactly match pixel ratio
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (h * w) 

    # linearly interpolate target and logit_t in the same way
    target_oh = make_one_hot(target, opt.n_cls)
    target_mix  = lam * target_oh + (1 - lam) * target_oh[rand_index]
    logit_t_mix = lam *   logit_t + (1 - lam) *   logit_t[rand_index]
    logit_s_mix = lam *   logit_s + (1 - lam) *   logit_s[rand_index]

    return input_mix, target_mix, logit_t_mix, logit_s_mix, rand_index, lam

def cutout(input, target=None, logit_t=None, logit_s=None, model_t=None, model_s=None, opt=None):
    r"""Refer to the official cutout impl.: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    """
    n_holes = 1 # Refer to https://github.com/uoguelph-mlrg/Cutout, default param
    if opt.dataset in ['cifar10']:
        length = 16 # Refer to Cutout paper
    if opt.dataset in ['cifar100']:
        length = 8 # Refer to Cutout paper
    elif opt.dataset in ['tinyimagenet']:
        length = 16 # Empirically set, referring to the CIFAR100 case
    elif opt.dataset in ['imagenet', 'imagenet100']:
        # The official Cutout does not work well on ImageNet (see https://github.com/uoguelph-mlrg/Cutout/issues/4)
        # Empirically set, referring to the CIFAR100 case: 224 / (32 / 8) = 56
        length = 56
    else:
        raise NotImplementedError
    _, _, h, w = input.size()
    mask = torch.ones((h, w)).cuda()
    for _ in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
    mask = mask.expand_as(input)
    input_mix = input * mask
    target_mix, logit_t_mix, logit_s_mix, rand_index, lam = None, None, None, None, None # placeholder to maintain interface
    return input_mix, target_mix, logit_t_mix, logit_s_mix, rand_index, lam

def masking(input, target=None, logit_t=None, logit_s=None, model_t=None, model_s=None, opt=None):
    r"""Similar interface to cutout.
    This implements the masking operation in MAE (https://arxiv.org/abs/2111.06377).
    """
    if opt.dataset in ['mnist', 'cifar10', 'cifar100', 'svhn']:
        img_size = 32
    elif opt.dataset in ['tinyimagenet']:
        img_size = 64
    elif opt.dataset in ['imagenet', 'imagenet100']:
        img_size = 224
    else:
        raise NotImplementedError
    n_patch = opt.n_patch 
    mask = torch.randperm(n_patch * n_patch) >= int(opt.mask_zero_ratio * n_patch * n_patch) # This presumes the input is square
    mask = mask.view(n_patch, n_patch) # [n_patch, n_patch]
    mask = mask.float().unsqueeze(dim=0).unsqueeze(dim=0) # [1, 1, n_patch, n_patch]
    mask = F.upsample_nearest(mask, scale_factor=img_size//n_patch).cuda()
    input_mix = input * mask
    target_mix, logit_t_mix, logit_s_mix, rand_index, lam = None, None, None, None, None # placeholder to maintain interface
    return input_mix, target_mix, logit_t_mix, logit_s_mix, rand_index, lam

def patchmix(input, target, logit_t, logit_s, model_t, model_s, opt):
    r""" A more advanced version of cutmix.
    """
    batch_size, c, h, w = input.size()
    
    # get random bounding box
    beta = 1.0
    lam = np.random.beta(beta, beta) # when beta=1, it degrades to the uniform dist
    if opt.ceiling_ratio != -1:
        # lam = (2 * ceiling_ratio - 1) * lam + (1 - ceiling_ratio) # (0, 1) -> (1-ceiling_ratio, ceiling_ratio)
        # lam = lam * ceiling_ratio                                 # (0, 1) -> (1-ceiling_ratio, ceiling_ratio)
        lam = opt.floor_ratio + (opt.ceiling_ratio - opt.floor_ratio) * lam     # (0, 1) -> (floor_ratio, ceiling_ratio)

    # get shuffle index
    rand_index = torch.randperm(batch_size).cuda()
    while 0 in (rand_index - torch.arange(batch_size).cuda()): # a small change from original cutmix impl.: use non-overlap rand_index
        rand_index = torch.randperm(input.size()[0]).cuda()

    # get new image
    patch_size = 4
    mask = torch.ones(1, 1, h//patch_size, w//patch_size).cuda()
    spatial_size = h//patch_size * w//patch_size
    num_zeros = int(spatial_size * lam)
    ix = np.random.permutation(spatial_size)[:num_zeros]
    mask = mask.flatten()
    mask[ix] = 0
    mask = mask.view(1, 1, h//patch_size, w//patch_size)
    mask = F.interpolate(mask, scale_factor=(patch_size, patch_size))
    input_mix = input * mask + input[rand_index] * (1 - mask)

    target_mix, logit_t_mix, logit_s_mix, rand_index, lam = None, None, None, None, None # placeholder to maintain interface
    return input_mix, target_mix, logit_t_mix, logit_s_mix, rand_index, lam

def strprint(x):
    logstr = ''
    for xi in x:
        logstr += '%3d ' % xi
    print(logstr)

def check_overlap(x, y):
    overlap = [i for i in x if i in y] 
    return float(len(overlap)) / float(len(x))

def get_entropy_batch(img, model_t, model_s, temp, cutmix_pick_criterion, logit_t_ori):
    '''Get entropy by model forwarding
    '''
    with torch.no_grad():
        if cutmix_pick_criterion == 'dissimilarity':
            logit_t_aug = model_t(img)
            num_aug, num_ori = logit_t_aug.size(0), logit_t_ori.size(0)
            
            logit_t_aug = logit_t_aug.data.cpu().numpy() # pt0.4.1 has no 'repeat_interleave' func, use numpy instead
            logit_t_aug = np.repeat(logit_t_aug, num_ori, axis=0) # shape: [num_aug x num_ori, num_classes]
            logit_t_aug = torch.from_numpy(logit_t_aug).cuda()
            logit_t_ori = logit_t_ori.repeat(num_aug, 1) # shape: [num_aug x num_ori, num_classes]
            prob_t_aug = F.softmax(logit_t_aug, dim=1)
            prob_t_ori = F.softmax(logit_t_ori, dim=1)
            
            l2_distance = F.mse_loss(prob_t_aug, prob_t_ori, reduction='none').sum(dim=1) # [num_aug x num_ori,]
            l2_distance = l2_distance.view([num_aug, num_ori])
            l2_distance, _ = torch.min(l2_distance, dim=1) # [num_aug]
            
            # --- check different metrics
            # l1_distance = F.l1_loss(prob_t_aug, prob_t_ori, reduction='none').sum(dim=1) # [num_aug x num_ori,]
            # l1_distance = l1_distance.view([num_aug, num_ori])
            # l1_distance, _ = torch.min(l1_distance, dim=1) # [num_aug]
            
            # kld = F.kl_div(prob_t_aug.log(), prob_t_ori, reduction='none').sum(dim=1)
            # kld = kld.view([num_aug, num_ori])
            # kld, _ = torch.min(kld, dim=1) # [num_aug]

            # _, index_l2 = l2_distance.sort()
            # _, index_l1 = l1_distance.sort()
            # _, index_kld = kld.sort()

            # strprint(index_l2[-64:])
            # strprint(index_l1[-64:])
            # strprint(index_kld[-64:])
            # print('overlap, l1 and l2 : %.4f' % check_overlap(index_l2[-64:],  index_l1[-64:]))
            # print('overlap, l1 and kld: %.4f' % check_overlap(index_l1[-64:], index_kld[-64:]))
            # print('overlap, l2 and kld: %.4f' % check_overlap(index_l2[-64:], index_kld[-64:]))
            # 
            # conclusion: overlap is large: 0.98+, i.e., only 1 or 2 of 64 choices are different
            # ---
            return l2_distance
        
        elif cutmix_pick_criterion == 'teacher_entropy':
            prob = model_t(img).softmax(dim=1) # [N, C]
            return (-prob * torch.log(prob)).sum(dim=1) # [N]
        
        elif cutmix_pick_criterion == 'student_entropy':
            prob = model_s(img).softmax(dim=1) # [N, C]
            return (-prob * torch.log(prob)).sum(dim=1) # [N]
            
        elif cutmix_pick_criterion == 'kld':
            prob_t     =     F.softmax(model_t(img) / temp, dim=1) # [N, C]
            log_prob_s = F.log_softmax(model_s(img) / temp, dim=1) # [N, C]
            kld = F.kl_div(log_prob_s, prob_t, reduction='none').sum(dim=1) # [N]
            return kld

def get_entropy_batch_interpolate(logit_t, logit_s, temp, cutmix_pick_criterion):
    '''Get entropy by interpolation instead of model forwarding
    '''
    if cutmix_pick_criterion == 'teacher_entropy':
        prob = logit_t.softmax(dim=1)
        return (-prob * torch.log(prob)).sum(dim=1)
    
    elif cutmix_pick_criterion == 'student_entropy':
        prob = logit_s.softmax(dim=1)
        return (-prob * torch.log(prob)).sum(dim=1)
    
    elif cutmix_pick_criterion == 'kld':
        prob_t     =     F.softmax(logit_t / temp, dim=1)
        log_prob_s = F.log_softmax(logit_s / temp, dim=1)
        kld = F.kl_div(log_prob_s, prob_t, reduction='none').sum(dim=1)
        return kld

# Pick large-entropy cutmix samples
def cutmix_pick(input, target, logit_t, logit_s, model_t, model_s, opt):
    # Run cutmix multi-times, collect the results into a pool
    input_mix_pool, target_mix_pool, logit_t_mix_pool,logit_s_mix_pool = [], [], [], []
    for _ in range(opt.mix_n_run):
        input_mix, target_mix, logit_t_mix, logit_s_mix, *_ = cutmix(input, target, logit_t, logit_s, model_t, model_s, opt)
        input_mix_pool.append(input_mix)
        target_mix_pool.append(target_mix)
        logit_t_mix_pool.append(logit_t_mix)
        logit_s_mix_pool.append(logit_s_mix)
    input_mix_pool   = torch.cat(input_mix_pool,   dim=0).cuda()
    target_mix_pool  = torch.cat(target_mix_pool,  dim=0).cuda()
    logit_t_mix_pool = torch.cat(logit_t_mix_pool, dim=0).cuda()
    logit_s_mix_pool = torch.cat(logit_s_mix_pool, dim=0).cuda()

    entropy_aug = get_entropy_batch(input_mix_pool, model_t, model_s, opt.kd_T, opt.cutmix_pick_criterion, logit_t)
    if opt.entropy_log != None and opt.total_step % 100 == 0:
        # --- check variance of predicted probability, wrt different images
        # prob_t_ori = F.softmax(pick_model[0](input) / opt.kd_T, dim=1)
        # prob_t_aug = F.softmax(pick_model[0](input_mix_pool) / opt.kd_T, dim=1)
        # print('original  images, prob variance: %.6f' % np.var(prob_t_ori.data.cpu().numpy()))
        # print('augmented images, prob variance: %.6f' % np.var(prob_t_aug.data.cpu().numpy()))
        # ---
        entropy_ori =get_entropy_batch(input, model_t, model_s, opt.kd_T, opt.cutmix_pick_criterion)
        entropy_ori = entropy_ori.data.cpu().numpy()
        entropy_aug = entropy_aug.data.cpu().numpy()
        # ratio = entropy_aug_sorted[-opt.n_pick:].mean() / entropy_aug_sorted.mean() # average KL div ratio
        opt.entropy_log.append([opt.total_step, entropy_aug, entropy_ori])

    # data picking
    _, index = entropy_aug.sort() # in ascending order
    n_pick = min(len(index), opt.n_pick)
    index = index[-n_pick:]
    
    rand_index, lam = 0, 0 # placeholder
    return input_mix_pool[index], target_mix_pool[index], logit_t_mix_pool[index], logit_s_mix_pool[index], rand_index, lam

# Apply picking to other DA methods
def DA_pick(input, target, logit_t, logit_s, model_t, model_s, opt):
    assert opt.DA_pick_base is not None
    
    # Run DA multi-times, collect the results into a pool
    input_mix_pool, target_mix_pool, logit_t_mix_pool,logit_s_mix_pool = [], [], [], []
    for _ in range(opt.mix_n_run):
        input_mix, target_mix, logit_t_mix, logit_s_mix, *_ = get_mixed_input(opt.DA_pick_base, input, target, logit_t, logit_s.detach(), model_t, model_s, opt)
        input_mix_pool.append(input_mix)
        target_mix_pool.append(target_mix)
        logit_t_mix_pool.append(logit_t_mix)
        logit_s_mix_pool.append(logit_s_mix)
    input_mix_pool   = torch.cat(input_mix_pool,   dim=0).cuda()
    
    if target_mix_pool[0] is not None:
        target_mix_pool = torch.cat(target_mix_pool,  dim=0).cuda()
    else:
        target_mix_pool = np.array([None] * input.size(0) * opt.mix_n_run)
    
    if logit_t_mix_pool[0] is not None:
        logit_t_mix_pool = torch.cat(logit_t_mix_pool, dim=0).cuda()
    else:
        logit_t_mix_pool = np.array([None] * input.size(0) * opt.mix_n_run)
    
    if logit_s_mix_pool[0] is not None:
        logit_s_mix_pool = torch.cat(logit_s_mix_pool, dim=0).cuda()
    else:
        logit_s_mix_pool = np.array([None] * input.size(0) * opt.mix_n_run)

    entropy_aug = get_entropy_batch(input_mix_pool, model_t, model_s, opt.kd_T, opt.cutmix_pick_criterion, logit_t)
    if opt.entropy_log != None and opt.total_step % 100 == 0:
        # --- check variance of predicted probability, wrt different images
        # prob_t_ori = F.softmax(pick_model[0](input) / opt.kd_T, dim=1)
        # prob_t_aug = F.softmax(pick_model[0](input_mix_pool) / opt.kd_T, dim=1)
        # print('original  images, prob variance: %.6f' % np.var(prob_t_ori.data.cpu().numpy()))
        # print('augmented images, prob variance: %.6f' % np.var(prob_t_aug.data.cpu().numpy()))
        # ---
        entropy_ori =get_entropy_batch(input, model_t, model_s, opt.kd_T, opt.cutmix_pick_criterion)
        entropy_ori = entropy_ori.data.cpu().numpy()
        entropy_aug = entropy_aug.data.cpu().numpy()
        # ratio = entropy_aug_sorted[-opt.n_pick:].mean() / entropy_aug_sorted.mean() # average KL div ratio
        opt.entropy_log.append([opt.total_step, entropy_aug, entropy_ori])

    # data picking
    _, index = entropy_aug.sort() # in ascending order
    n_pick = min(len(index), opt.n_pick)
    index = index[-n_pick:]
    
    rand_index, lam = 0, 0 # placeholder
    # return input_mix_pool[index], target_mix_pool[index], logit_t_mix_pool[index], logit_s_mix_pool[index], rand_index, lam
    return input_mix_pool[index], None, None, None, rand_index, lam


def get_mixed_input(mix_mode, input, target, logit_t, logit_s, model_t, model_s, opt):
    r"""Return:
        - input_mix: the mixed input
        - logit_t_mix: mixing the logit_t of the input with the same way as mixing the input
        - target_mix: mixing the target of the input with the same way as mixing the input
    """    
    if mix_mode in ['manifold_mixup']: # old mixup impel. will be removed
        mixup_alpha = 1.0
        if mix_mode == 'manifold_mixup':
            mixup_alpha = 2.0 # refer to official manifold mixup imple: https://github.com/vikasverma1077/manifold_mixup/tree/master/supervised
        logit_s, logit_s_input_mix, target_mix, logit_t_mix, input_mix = model_s.forward_mixup(input, 
                                                        mix_mode=mix_mode,
                                                        mixup_alpha=mixup_alpha,
                                                        target=target,
                                                        logit_t=logit_t,
                                                        mixup_output='logit',
                                                        n_run=opt.mix_n_run,
                                                        cut_size=opt.cut_size)

    # Refers to the official pytorch usage of AutoAugment: 
    # https://pytorch.org/vision/master/transforms.html#torchvision.transforms.AutoAugment
    elif mix_mode in ['autoaugment']:
        from torchvision.transforms.autoaugment import AutoAugment
        from torchvision.transforms.autoaugment import AutoAugmentPolicy
        from utils import denormalize_image, normalize_image
        if opt.dataset in ['imagenet', 'imagenet100']:
            policy = AutoAugmentPolicy.IMAGENET
            from dataset.imagenet import MEAN, STD
        elif opt.dataset in ['cifar100', 'cifar10']:
            policy = AutoAugmentPolicy.CIFAR10
            from dataset.cifar100 import MEAN, STD
        elif opt.dataset in ['tinyimagenet']:
            policy = AutoAugmentPolicy.IMAGENET
            from dataset.tinyimagenet import MEAN, STD
        autoaug = AutoAugment(policy=policy)
        # ******************************************
        input = denormalize_image(input, mean=MEAN, std=STD)
        input = (input.mul(255)).type(torch.uint8) # Autoaugment in pytorch demands the input is int8 if the input is a tensor
        input_mix = autoaug(input).type(torch.float32).div(255)
        input_mix = normalize_image(input_mix, mean=MEAN, std=STD)
        # ******************************************
        rand_index = lam = target_mix = logit_t_mix = logit_s_mix = None # maintain interface compatibility

    elif mix_mode in ['identity']:
        input_mix = input.clone()
        rand_index = lam = target_mix = logit_t_mix = logit_s_mix = None # maintain interface compatibility
    
    elif mix_mode in ['crop', 'flip', 'crop+flip']:
        from torchvision.transforms import RandomHorizontalFlip
        from torchvision.transforms import RandomCrop
        from utils import denormalize_image, normalize_image
        flip = RandomHorizontalFlip()
        if opt.dataset in ['cifar100', 'cifar10']:
            img_size, padding = 32, 4
            from dataset.cifar100 import MEAN, STD
        
        elif opt.dataset in ['tinyimagenet']:
            img_size, padding = 64, 8
            from dataset.tinyimagenet import MEAN, STD
        
        elif opt.dataset in ['imagenet', 'imagenet100']:
            img_size, padding = 224, 28 # Referring to CIFAR, 224 / (32 /4) = 28
            # The crop in training used by ImageNet is actually ResizedCrop. Tried, does not work yet.
            from dataset.imagenet import MEAN, STD
        
        crop = RandomCrop(img_size, padding=padding)
        # ******************************************
        input = denormalize_image(input, mean=MEAN, std=STD)
        input = (input.mul(255)).type(torch.uint8)
        
        if 'crop' in opt.mix_mode:
            input = crop(input)
        if 'flip' in opt.mix_mode:
            input = flip(input)
        
        input_mix = input.type(torch.float32).div(255)
        input_mix = normalize_image(input_mix, mean=MEAN, std=STD)
        # ******************************************
        rand_index = lam = target_mix = logit_t_mix = logit_s_mix = None # maintain interface compatibility
    
    elif mix_mode in ['same']:
        input_mix = input[0].repeat(input.size(0), 1, 1, 1) # [C, H, W] -> [batch_size, C, H, W]
        rand_index = lam = target_mix = logit_t_mix = logit_s_mix = None # maintain interface compatibility

    else: # TODO: Fully check the opt.mix_mode
        mix_func = eval(mix_mode)
        input_mix, target_mix, logit_t_mix, logit_s_mix, rand_index, lam = mix_func(input, target, logit_t, logit_s, model_t, model_s, opt)

        if opt.check_cutmix_label:
            assert opt.dataset == 'imagenet'
            check_cutmix_label_imagenet(input_mix, target_mix, model_t, opt) # @mst-TODO: within, model_t(input_mix) can be saved.

    return input_mix, target_mix, logit_t_mix, logit_s_mix, rand_index, lam

def check_cutmix_label_imagenet(input_mix, target_mix, model_t, opt):
    # init
    if not hasattr(opt, 'n_disagree_img'): 
        opt.n_disagree_img = 0
        opt.n_total_img = 0
        with open('tools/imagenet_synsets.txt', 'r') as f: # ImageNet synsets to get the semantic name of each class
            synsets = f.readlines()
        synsets = [x.strip() for x in synsets]
        splits = [line.split(' ') for line in synsets]
        opt.key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}
        with open('tools/imagenet_classes.txt', 'r') as f:
            class_id_to_key = f.readlines()
        opt.class_id_to_key = [x.strip() for x in class_id_to_key]

    # denormalize input_mix for later saving
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225] # imagenet mean and std
    input_mix_denormalized = denormalize_image(input_mix, mean, std)
    
    with torch.no_grad():
        out_t = model_t(input_mix)
        prob_t = out_t.softmax(dim=1)
        target_t = out_t.argmax(dim=1)
        target_mix_as_label = target_mix.argmax(dim=1) # target_mix is like [0.34, 0.66, 0], target_mix_as_label is like 1.
        opt.n_total_img += input_mix.size(0)
        for i in range(target_t.size(0)):
            y_t = target_t[i].item()
            y_mix = target_mix_as_label[i].item()
            if y_t != y_mix: # teacher disagrees with cutmix about the label of mixed input
                opt.n_disagree_img += 1
                save_img_path = '%s/%d_mixlabel%s_teacherlabel%s.png' % (opt.logger.gen_img_path, opt.n_disagree_img, y_mix, y_t)
                vutils.save_image(input_mix_denormalized[i].cpu().data, save_img_path)
                
                # teacher's predicted prob:
                p_t, index_t = torch.sort(prob_t[i], descending=True)
                top5_class_names = []
                top5_prob = p_t[:5]
                for c in index_t[:5]:
                    classname = opt.key_to_classname[opt.class_id_to_key[c]]
                    top5_class_names.append(classname)

                # cutmix's prob:
                p_cutmix, index_cutmix = torch.sort(target_mix[i], descending=True)
                top2_class_names = []
                top2_prob = p_cutmix[:2]
                for c in index_cutmix[:2]:
                    classname = opt.key_to_classname[opt.class_id_to_key[c]]
                    top2_class_names.append(classname)
                
                save_txt_path = '%s/%d_prob.txt' % (opt.logger.gen_img_path, opt.n_disagree_img)
                with open(save_txt_path, 'w+') as f:
                    print(top5_prob.data.cpu().numpy(), file=f)
                    print(top5_class_names, file=f)
                    print(top2_prob.data.cpu().numpy(), file=f)
                    print(top2_class_names, file=f)
                    print('label disgreed on %d/%d images' % (opt.n_disagree_img, opt.n_total_img), file=f)    

def cross_entropy(logit, logit_gt, input_logit=True, reduction='sum'):
    r"""Calculate the cross-entropy between the distribution of <logit> and the distribution of <logit_gt>. 
    """
    if input_logit:
        p, p_gt = F.softmax(logit, dim=-1), F.softmax(logit_gt, dim=-1) # [batch_size, num_classes] 
    else:
        p, p_gt = logit, logit_gt
    ce = -p_gt * torch.log(p)
    if reduction in ['sum']:
        return ce.sum()
    elif reduction in ['batch_mean']:
        return ce / p.size(0)
    elif reduction in ['mean']:
        return ce / p.numel()
    elif reduction in ['none']:
        return ce
    else:
        raise NotImplemented

def kld(logit, logit_gt, input_logit=True, reduction='sum'):
    r"""Calculate the KL divergence between the distribution of <logit> and the distribution of <logit_gt>. 
    """
    p = F.log_softmax(logit, dim=-1)
    q = F.softmax(logit_gt, dim=-1)
    loss = F.kl_div(p, q, reduction=reduction)
    return loss