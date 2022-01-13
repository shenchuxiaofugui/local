import torch.nn.functional as F
from utils.dice_score import dice_loss


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_loss(output, target):
    sum = 0
    for i in range(10):
        out = output[:, 2*i:(2*i+2), :, :]
        label = target[:, i, :, :].squeeze()
        a = F.cross_entropy(out, label)
        # + dice_loss(F.softmax(out, dim=1).float(),
        #             label.float(),
        #             multiclass=False)
        sum += a
    return sum
    # return F.cross_entropy(output, target) \
    #     + dice_loss(F.softmax(output, dim=1).float(),
    #                 target.float(),
    #                 multiclass=False)
