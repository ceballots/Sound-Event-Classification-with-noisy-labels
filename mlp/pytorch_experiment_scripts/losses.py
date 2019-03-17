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
#        print('MAE W/O EPS')
        p = torch.zeros(targets.shape[0],num_classes)
        targets = p.scatter_(1, y_no_cuda.view(-1,1),1)
        targets = targets.type(torch.cuda.FloatTensor)
        pred_probs = F.softmax(pred,dim=-1)
        loss = nn.L1Loss()
        return loss(pred_probs,targets)
        
    pred_probs = F.softmax(pred,dim=-1)
    pred_probs = pred_probs.to(device)
    pred = F.log_softmax(pred, dim=-1)
    targets_int =  y_no_cuda
    idx_manual = np.nonzero(array_manual_label==1)[0]
    if eps_smoothing > 0:
        eps_per_class = eps_smoothing / num_classes
        
        p = torch.zeros(y_no_cuda.shape[0],num_classes).fill_(eps_per_class)
        targets = p.scatter_(1, y_no_cuda.view(-1,1),1-eps_smoothing)
                
        if (consider_manual and
            array_manual_label.shape[0] > 0 and
            len(idx_manual) > 0):
            targets = consider_manual_labeled(targets,targets_int,array_manual_label,num_classes)
        
        targets = targets.to(device)
        pred = pred.to(device)
        if loss_function == 'CCE':
            loss = -(targets * pred)
            loss = loss.sum(dim=1)
            loss = loss.mean()
            return loss
        elif loss_function == 'MAE':
#            print('mae w/ eps')
#            print(pred_probs)
 #           print(targets)
            return nn.L1Loss(pred_probs,targets)
                

def consider_manual_labeled(targets_one_hot,targets,array_manual_label,num_classes):
    """
    Function that does not smooth manual verified labels
    
    :torch targets:
    :array array_manual_label:
    :param num_classes:
    """
    
    count = 0
    idx_manual = np.nonzero(array_manual_label==1)[0]
    targets = targets[idx_manual]
    idx_manual = torch.from_numpy(idx_manual)
    idx_manual = idx_manual.long()
    subs = torch.zeros(targets.shape[0],num_classes).scatter_(1,targets.view(-1,1),1)
    targets_one_hot[idx_manual] =  subs
    return targets_one_hot
"""
    for index in range(targets.shape[0]):
        if array_manual_label[index] == 1:

            targets_one_hot[index] = subs[count]
            count = count + 1
    
    """
  
