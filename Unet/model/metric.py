import torch


def accuracy(output, target):
    with torch.no_grad():
        out = output[:, :2, :, :]
        pred = torch.argmax(out, dim=1).unsqueeze(1)
        for i in range(1, 10):
            out = output[:, 2 * i:(2 * i + 2), :, :]
            pred1 = torch.argmax(out, dim=1).unsqueeze(1)
            pred = torch.cat((pred, pred1), dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        print(pred.shape, target.shape)
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
