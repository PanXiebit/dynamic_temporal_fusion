

from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch import Tensor


class CtcLoss(_Loss):
    def __init__(self, opts, blank_id, device, reduction="none"):
        super(CtcLoss, self).__init__()
        self.ctcloss = nn.CTCLoss(blank=blank_id, reduction=reduction)
        self.device = device

    def forward(self, model, samples):
        video = samples["data"]
        len_video = samples["len_data"]
        label = samples["label"]            # "(sum(target_lengths))"
        len_label = samples["len_label"]
        logits, _ = model(video, len_video)
        len_video /= 4
        logits = logits.permute(1, 0, 2)
        log_probs = logits.log_softmax(-1)   # T x N x C
        loss = self.ctcloss(log_probs.cpu(), label.cpu(), len_video.cpu(), len_label.cpu())
        # loss = loss.mean()
        loss = loss.sum() / video.size(0)
        return loss.to(self.device)