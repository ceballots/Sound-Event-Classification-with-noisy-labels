"""Dropout scheduler.
This module contains classes implementing schedulers which control the
evolution of learning rule hyperparameters (such as dropout) over a
training run.
"""



import numpy as np
import math

class DropoutScheduler(object):
    "Dropout scheduler"
    def __init__(self, min_dropout, max_dropout, total_num_epochs = 100): 
                 
                 #total_iters_per_period, max_learning_rate_discount_factor,
                 #min_learning_rate_discount_factor = 1.,period_iteration_expansion_factor = 1):
        """
        Instantiates a new cosine annealing with warm restarts learning rate scheduler
        :param min_dropout: The minimum dropout rate the scheduler can assign
        :param max_dropout: The maximum dropout rate the scheduler can assign
        :param total_num_epochs: The number of epochs 
        """
        
        self.min_dropout = min_dropout
        self.max_dropout = max_dropout
        self.total_num_epochs = total_num_epochs
        
    def UpdateDropout(self, epoch_number):
        """
        Update Dropout according to:
        
         dropout(epoch_number) = min_dropout + 
                                 (max_dropout - min_dropout)·sin((epoch_number/total_number_epochs)(PI/2))
         
         To be run at the beginning of each epoch
         
         Args:
         
            :param epoch_number: current epoch_number (int)       
        
        """
        dropout_updated = (self.min_dropout + 
                           0.5*(self.max_dropout - self.min_dropout)*
                           (1 - math.cos(math.pi*(epoch_number/self.total_num_epochs))))
        
        #dropout_updated = (self.min_dropout + (self.max_dropout - self.min_dropout)* 
        #                   math.sin((epoch_number/self.total_num_epochs)*math.pi/2))
        
        return dropout_updated