from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import math
import torch

def kd_loss(student_scores, teacher_scores, temp=1):
    '''Knowledge distillation loss: soft target
    '''
    p = F.log_softmax(student_scores / temp, dim=1)
    q =     F.softmax(teacher_scores / temp, dim=1)
    l_kl = F.kl_div(p, q, reduction='none').sum(dim=1) * temp * temp # shape: [batch_size]
    return l_kl

def kld_to_weight(kld, max_min_ratio): # kld is a list tensor
    if max_min_ratio == 1:
        return torch.ones_like(kld) / kld.size(0)
    tau = (kld.max() - kld.min()) / math.log(max_min_ratio)
    weight = F.softmax(kld / tau)
    return weight

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T_T, T_S):
        super(DistillKL, self).__init__()
        self.T_T = T_T
        self.T_S = T_S

    def forward(self, y_s, y_t, prob_ratio=0):
        if prob_ratio != 0: # kd-Huan
            sorted_y_t = y_t.sort(dim=1)[0]
            tau_ = (sorted_y_t[:, -2] - sorted_y_t[:, 0]) / math.log(prob_ratio)
            tau_ = tau_.unsqueeze(dim=1).detach() # [batch_size, 1], temp for each example
            p_s = F.log_softmax(y_s / tau_, dim=1)
            p_t = F.softmax(y_t / tau_, dim=1)
            loss = (F.kl_div(p_s, p_t, reduction='none') * (tau_ ** 2)).sum() / y_s.shape[0]
            return loss, tau_
        else:
            p_s = F.log_softmax(y_s/self.T_S, dim=1)
            p_t =     F.softmax(y_t/self.T_T, dim=1)
            loss = F.kl_div(p_s, p_t, size_average=False) * (self.T_S ** 2) / y_s.shape[0]
            return loss
    
    def forward_reweight(self, y_s, y_t, y_t2, max_min_ratio):
        kd_loss_T2_S = kd_loss(y_s, y_t2, temp=self.T_S) # shape: [batch_size], kld loss for each example
        kld_T1_T2 = kd_loss(y_t, y_t2)
        weight = kld_to_weight(kld_T1_T2, max_min_ratio=max_min_ratio) * y_s.size(0)
        weighted_kd_loss_T2_S = weight * kd_loss_T2_S # shape: [batch_size]

        # # check
        # tmp = '' 
        # for w in weight:
        #     tmp += '%.3f ' % w.item()
        # print(tmp)
        # print('%.4f  %.4f  %.4f' % (weight.max().item(), weight.min().item(), weight.mean().item()))
        # print('original_loss: max: %.4f  min: %.4f  mean: %.4f' % (kd_loss_T2_S.max().item(), kd_loss_T2_S.min().item(), kd_loss_T2_S.mean().item()))
        # print('     now_loss: max: %.4f  min: %.4f  mean: %.4f' % (weighted_kd_loss_T2_S.max().item(), weighted_kd_loss_T2_S.min().item(), weighted_kd_loss_T2_S.mean().item()))
        
        return weighted_kd_loss_T2_S.mean()