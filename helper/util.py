from __future__ import print_function

import torch
import numpy as np


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def get_teacher_name(model_path):
    """parse teacher name"""
    # segments = model_path.split('/')[-2].split('_')
    # if segments[0] != 'wrn':
    #     return segments[0]
    # else:
    #     return segments[0] + '_' + segments[1] + '_' + segments[2]
    return model_path.split('/')[-2].split('_vanilla')[0]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) # Because of pytorch new versions, this does not work anymore (pt1.3 is okay, pt1.9 not okay).
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# kd-Huan: 
class CeilingRatioScheduler(object):
    """ceiling_ratio scheduler for cutmix_pick
    """
    def __init__(self, decay_schedule):
        self.decay_schedule = {}
        for k, v in decay_schedule.items(): # a dict, example: {"0":0.001, "30":0.00001, "45":0.000001}
            self.decay_schedule[int(k)] = v
    def get_ceiling_ratio(self, e):
        epochs = list(self.decay_schedule.keys())
        epochs = sorted(epochs) # example: [0, 30, 45]
        ratio = self.decay_schedule[epochs[-1]]
        for i in range(len(epochs) - 1):
            if epochs[i] <= e < epochs[i+1]:
                ratio = self.decay_schedule[epochs[i]]
                break
        return ratio
        
if __name__ == '__main__':

    pass
