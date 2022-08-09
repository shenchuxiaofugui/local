import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_loss(output, target):
    return F.cross_entropy(output, target)


def BCEWithLogitsLoss(output, target):
    output = torch.argmax(output, dim=1)
    # output = output.type(torch.FloatTensor)
    # target = target.type(torch.FloatTensor)
    output = output.to(torch.float32)
    target = target.to(torch.float32)
    return F.binary_cross_entropy_with_logits(output, target)
