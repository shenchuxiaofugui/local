import torch
import torchmetrics
from sklearn.metrics import roc_auc_score


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def AUC(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        if len(set(target)) == 1:
            correct = 0.5
        #print('heheheheh', pred, target)
        else:
            correct = roc_auc_score(target, pred)
    return correct

