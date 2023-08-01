import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')


    def forward(self, inputs, targets):
        # targets = targets.to(torch.float32)
        loss = self.criterion(inputs, targets)
        loss = torch.sqrt(loss)
        return loss