import torch
from torch import nn
from .memory import ContrastMemory

eps = 1e-7


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        if opt.embed == 'deeper':
            EmbedNet = Embed_Deeper
        elif opt.embed == 'null':
            EmbedNet = Embed_Null
        else:
            EmbedNet = Embed
        self.embed_s = EmbedNet(opt.s_dim, opt.feat_dim)
        self.embed_t = EmbedNet(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m) # feat_dim: 128
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)
        self.opt = opt
        # --- kd-Huan: save the loss for check. This is just exploring code, will be removed.
        if opt.save_crd_loss:
            self.loss_log_s = open(opt.logger_my.log_path + '/crd_loss_log_s.txt', 'a+')
            self.loss_log_t = open(opt.logger_my.log_path + '/crd_loss_log_t.txt', 'a+')
        else:
            self.loss_log_s = None
            self.loss_log_t = None
        # ---

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx) # out_s, out_t: the prob of P(i | v) in the meomry bank paper
        s_loss = self.criterion_s(out_s, self.loss_log_s) # out_s: when student's embeddings are the anchor 
        t_loss = self.criterion_t(out_t, self.loss_log_t) # out_t: when teacher's embeddings are the anchor
        loss = s_loss + t_loss
        return loss
    
    def update_memory_bank(self, f_s, f_t, idx):
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        self.contrast(f_s, f_t, idx, only_update_memory_bank=True)
        return
        
class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data
        # print("n_data = %d" % self.n_data)

    def forward(self, x, loss_log=None):
        # x size example: torch.Size([64, 16385, 1])
        bsz = x.shape[0]
        m = x.size(1) - 1 # number of negatives

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0) # dim=1, slice 0
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_() # Get h, Eq.19, Eq.10 in CRD, shape: torch.Size([64, 1])

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m) # dim=1, index starting from 1, slice out m elements
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_() # Eq.19, Eq.10 in CRD, shape: torch.Size([64, 16384, 1])

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz # averaging or obtain expectation

        # --- kd-Huan
        if loss_log:
            pos_loss = -log_D1.sum(0).item() / bsz
            neg_loss = -log_D0.view(-1, 1).sum(0).item() / bsz
            print("%.2f  %.2f  %.10f  %.10f" % (pos_loss, neg_loss, x[:, 0].mean().item(), x[:, 1:].mean().item()), file=loss_log, flush=True)
        # ---
        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1) 
        x = self.linear(x)
        x = self.l2norm(x)
        return x

class Embed_Deeper(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed_Deeper, self).__init__()
        n_dim = 256
        self.linear1 = nn.Linear(dim_in,  n_dim)
        self.linear2 = nn.Linear(n_dim, dim_out)
        self.relu = nn.ReLU()
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1) 
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.l2norm(x)
        return x

class Embed_Null(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed_Null, self).__init__()
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l2norm(x)
        return x

class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm) 
        return out
