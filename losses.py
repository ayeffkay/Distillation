import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class CELossWithT(nn.Module):
    def __init__(self, T: int=1):
        super().__init__()
        self.T = T

    def set_temp(self, T: int=1):
        self.T = T

    def forward(self, logit: torch.FloatTensor, label: torch.LongTensor, *args):
        label = label.view(-1)
        loss = F.cross_entropy(logit / self.T, label)
        
        return loss


class CEWithTeacherLogits(nn.Module):
    def __init__(self, T: int=1):
        super().__init__()
        self.T = T

    def set_temp(self, T: int=1):
        self.T = T

    def forward(self, s: torch.FloatTensor, t: torch.FloatTensor, *args):
        """"
            s -- student logit
            t -- teacher logit
        """
        soft_targets_s = F.softmax(s / self.T, dim=1)
        soft_targets_t = F.softmax(t / self.T, dim=1)
        loss = torch.sum(-soft_targets_t * torch.log(soft_targets_s), dim=1).mean()
        return loss
    
    
class DistillationLoss(nn.Module):
    def __init__(self, alpha: float=2, T: int=1):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.ce_loss = CEWithTeacherLogits(T)

    def set_temp(self, T: int=1):
        self.T = T
        self.ce_loss.set_temp(T)

    def forward(self, s: torch.FloatTensor, targets: torch.LongTensor, 
                t: torch.FloatTensor):
        """
            s -- student logit
            t -- teacher logit
            targets -- real labels
        """
        loss = self.alpha * self.ce_loss(s, t) + 1 / self.alpha * F.cross_entropy(s, targets.view(-1))
        return loss
    
