import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def lq_loss(y_true,y_pred,q=.8):
    """
    Implementation of loss function presented in:
     Zhilu Zhang and Mert R. Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with
     Noisy Labels", 2018
    https://arxiv.org/pdf/1805.07836.pdf
    
    :param y_true: true label
    :param y_pred: predicted label
    :param q: q param 0.8 by default
    :return: loss
    """
    _loss = torch.max(y_pred * y_truep)

    # compute the Lq loss between the one-hot encoded label and the prediction
    _loss = (1 - (_loss + 10 ** (-8)) ** _q) / _q

    return _loss

def loss_function(pred,targets,y_no_cuda,num_classes,device,eps_smoothing=0,loss_function='CCE',array_manual_label=None,consider_manual = False):
    """
    cross entropy loss with label smoothing https://arxiv.org/abs/1512.00567   
    Extended: array_manual_label indicates if smooth label is applied to that data entry
    
    :pred torch.FloatTensor: perdictions
    :targets torch.Long: Ground truth labels
    :num_classes param: number of classes
    :batch_size param:
    :eps_smoothing: [0,1] smoothing parameter
    :loss_function: CCE or MAE
    :array_manual_label: binary array to indicate if smooth label is applied to that data entry
    :consider_manual: flag
    """
    
    if (eps_smoothing == 0 and
       loss_function == 'CCE'):
        return F.cross_entropy(pred, targets)
    
    elif (eps_smoothing == 0 and
         loss_function == 'MAE'):
        print('MEA is not implemented yet')
        return 0
        
    
    pred = F.log_softmax(pred, dim=-1)
    
    if eps_smoothing > 0:
        eps_per_class = eps_smoothing / num_classes
        
        p = torch.zeros(y_no_cuda.shape[0],num_classes).fill_(eps_per_class)
        targets = p.scatter_(1, y_no_cuda.view(-1,1),1-eps_smoothing)
                
        if (consider_manual and
            len(array_manual_label) > 0):
            
            targets = consider_manual_labeled(targets,array_manual_label,num_classes)

        targets = targets.to(device)
        pred = pred.to(device)
        loss = -(targets * pred)
        
        loss = loss.sum(dim=1)
        
        loss = loss.mean()
        return loss

def consider_manual_labeled(targets,array_manual_label,num_classes):
    """
    Function that does not smooth manual verified labels
    
    :torch targets:
    :array array_manual_label:
    :param num_classes:
    """
    
    count = 0
    idx_manual = np.nonzero(array_manual_label==1)[0]
    
    idx_manual = torch.from_numpy(idx_manual)
    idx_manual = idx_manual.long()
    
    subs = torch.zeros(idx_manual.shape[0],num_classes).scatter_(1,idx_manual.view(-1,1),1)
    for index in range(targets.shape[0]):
        if array_manual_label[index] == 1:
            targets[index] = subs[count]
            count = count + 1
    
    return targets
    
