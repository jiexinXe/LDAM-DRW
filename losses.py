import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RLHybridSPFocalLoss(nn.Module):
    def __init__(self, num_classes, head_classes,
                 gamma0=2.0, alpha_r=0.1, tau=0.2,
                 lambda_max=1.0, total_epochs=200,
                 ema_decay=0.9, rl_eta=1.0, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.head_classes = set(head_classes)
        self.gamma0 = gamma0
        self.alpha_r = alpha_r
        self.tau = tau
        self.lambda_max = lambda_max
        self.total_epochs = total_epochs
        self.ema_decay = ema_decay
        self.rl_eta = rl_eta
        self.eps = eps

        self.register_buffer('class_loss_avg', torch.zeros(num_classes))
        self.register_buffer('class_entropy_avg', torch.zeros(num_classes))
        self.register_buffer('class_rl_weight', torch.ones(num_classes))

    def forward(self, logits, targets, epoch):
        N, C = logits.shape
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.view(-1,1)).squeeze(1)
        loss_ce = -torch.log(p_t + self.eps)

        # —— 更新类级 EMA 交叉熵 ——
        for c in torch.unique(targets):
            mask = (targets == c)
            self.class_loss_avg[c] = (
                self.ema_decay * self.class_loss_avg[c]
                + (1 - self.ema_decay) * loss_ce[mask].mean().detach()
            )

        # —— 计算样本熵并更新类级 EMA 熵 ——
        ent = -(probs * torch.log(probs + self.eps)).sum(dim=1)
        for c in torch.unique(targets):
            mask = (targets == c)
            self.class_entropy_avg[c] = (
                self.ema_decay * self.class_entropy_avg[c]
                + (1 - self.ema_decay) * ent[mask].mean().detach()
            )

        # —— 熵平衡权重（Inverse-Entropy） ——
        inv_h = 1.0 / (self.class_entropy_avg + self.eps)
        w_ent = inv_h / inv_h.sum()

        # —— Self-Paced Threshold ——
        lam = self.lambda_max * min(1.0, epoch / self.total_epochs)

        # —— 自适应 γ ——
        avg_c = self.class_loss_avg[targets]
        gamma_i = self.gamma0 * (loss_ce / (avg_c + self.eps))

        # —— Self-Paced 样本权重 ——
        w_sp = torch.ones_like(loss_ce)
        w_sp = torch.where(loss_ce < lam, torch.zeros_like(w_sp), w_sp)
        head_mask = torch.tensor([t in self.head_classes for t in targets],
                                 device=logits.device)
        redundant = (loss_ce < self.tau) & (loss_ce >= lam) & head_mask
        w_sp = torch.where(redundant, self.alpha_r * torch.ones_like(w_sp), w_sp)

        # —— 组合最终 loss ——
        # focal = (1 - p_t) ** gamma_i
        # w_rl = self.class_rl_weight[targets]
        # loss = w_sp * w_ent[targets] * w_rl * focal * loss_ce
        # return loss.mean()
        # —— 组合最终 loss（对数求和方式合并权重） ——
        focal = (1 - p_t) ** gamma_i
        w_rl = self.class_rl_weight[targets]

        # 避免 log(0)，对每个权重加上 eps
        log_w_sp = torch.log(w_sp + self.eps)
        log_w_ent = torch.log(w_ent[targets] + self.eps)
        log_w_rl = torch.log(w_rl + self.eps)

        # 三个权重在对数域相加，再回到原空间
        w_comb = torch.exp(log_w_sp + log_w_ent + log_w_rl)
        # （可选）保持整体权重规模不变：
        # w_comb = w_comb / (w_comb.mean().detach() + self.eps)

        loss = w_comb * focal * loss_ce
        return loss.mean()

    @torch.no_grad()
    def update_rl_weights(self, val_acc, prev_val_acc):
        # val_acc, prev_val_acc 都是长度 num_classes 的 list/array
        for c in range(self.num_classes):
            delta = max(0.0, val_acc[c] - prev_val_acc[c])
            self.class_rl_weight[c] *= math.exp(self.rl_eta * delta)
        # 归一化
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