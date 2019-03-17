import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

def mixup_data(x, y,y_no_cuda,num_classes, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    p = torch.zeros(y.shape[0],num_classes)

    y = p.scatter_(1, y_no_cuda.view(-1,1),1)
    y = y.type(torch.FloatTensor)

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b,y, lam

def mixup_criterion(pred, y_a, y_b, lam,device):
    criterion = nn.CrossEntropyLoss()
    y_a=y_a.to(
            device=device)
    y_b=y_b.to(
            device=device)
    pred = F.log_softmax(pred, dim=-1)
    loss=(lam *(pred*y_a) +  (1 - lam) * (pred*y_b))
    loss = -loss
    loss = loss.sum(dim=1)
        
    loss = loss.mean()
    return loss
