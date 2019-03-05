import torch
import torch.nn.functional as F



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
    