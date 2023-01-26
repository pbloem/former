

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import torch
import torch.nn.functional as F
from IPython import embed

def accuracy_v1(preds, labels, top=[1,5]):
    """Compute the precision@k for the specified values of k"""
    correct = [0] * len(top)
    _, labels_pred = torch.sort(preds, dim=1, descending=True)
    for idx, label_pred in labels_pred:
        result = (label_pred == labels[idx])
        j = 0
        for i in range(top[-1]):
            while i-1 > top[j]:
                j += 1
            if result[i] == 1:
                while j < len(top):
                    correct[j] += (100.0 / len(preds))
                # end the loop
                break
    return correct


def accuracy_v2(preds, labels, top=[1,5]):
    """Compute the precision@k for the specified values of k"""
    result = []
    maxk = max(top)
    batch_size = preds.size(0)

    _, pred = preds.topk(maxk, 1, True, True)
    pred = pred.t() # pred[k-1] stores the k-th predicted label for all samples in the batch.
    correct = pred.eq(labels.view(1,-1).expand_as(pred))

    for k in top:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))

    return result
