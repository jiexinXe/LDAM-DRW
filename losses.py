import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RLRedundancyLoss(nn.Module):
    """
    仅保留冗余样本降权 + RL 动态类权重，并使用加权和融合这两种权重。
    Loss = (alpha_comb * w_sp + (1 - alpha_comb) * w_rl) * CE
    """
    def __init__(self, num_classes, head_classes,
                 alpha_r=0.1, tau=0.2, rl_eta=0.5,
                 alpha_comb=0.5, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.head_classes = set(head_classes)
        self.alpha_r = alpha_r
        self.tau = tau
        self.rl_eta = rl_eta
        self.alpha_comb = alpha_comb
        self.eps = eps
        self.register_buffer('class_rl_weight', torch.ones(num_classes))

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.view(-1,1)).squeeze(1)
        loss_ce = -torch.log(p_t + self.eps)

        # 1) 冗余样本降权
        w_sp = torch.ones_like(loss_ce)
        head_mask = torch.tensor([t in self.head_classes for t in targets], device=logits.device)
        redundant = (loss_ce < self.tau) & head_mask
        w_sp = torch.where(redundant, self.alpha_r, w_sp)

        # 2) RL 动态类权重
        w_rl = self.class_rl_weight[targets]

        # 3) 加权和融合
        w_comb = self.alpha_comb * w_sp + (1 - self.alpha_comb) * w_rl

        # 最终 Loss
        loss = w_comb * loss_ce
        return loss.mean()

    @torch.no_grad()
    def update_rl_weights(self, val_acc, prev_val_acc):
        for c in range(self.num_classes):
            delta = max(0.0, val_acc[c] - prev_val_acc[c])
            self.class_rl_weight[c] *= math.exp(self.rl_eta * delta)
        # 归一化到 sum = num_classes
        self.class_rl_weight *= (self.num_classes / self.class_rl_weight.sum())

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)