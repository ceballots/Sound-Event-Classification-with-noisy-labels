from torch import nn
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import tqdm
import os



class FCCNetwork(nn.Module):
    def __init__(self, input_shape, num_output_classes, num_filters, num_layers, use_bias=False):
        """
        Initializes a fully connected network similar to the ones implemented previously in the MLP package.
        :param input_shape: The shape of the inputs going in to the network.
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every fcc layer.
        :param num_layers: Number of fcc layers (excluding dim reduction stages)
        :param use_bias: Whether our fcc layers will use a bias.
        """
        super(FCCNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()
        

    def build_module(self):
        print("Building basic block of FCCNetwork using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))

        out = x
        out = out.view(out.shape[0], -1)
                         # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim
        
        #num_filters = [24,48,48]
        
        print(type(self.num_filters))

        for i in range(self.num_layers):
            self.layer_dict['fcc_{}'.format(i)] = nn.Linear(in_features=out.shape[1],  # initialize a fcc layer
                                                            out_features=self.num_filters[i],
                                                            bias=self.use_bias)

            out = self.layer_dict['fcc_{}'.format(i)](out)  # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        self.logits_linear_layer = nn.Linear(in_features=out.shape[1],  # initialize the prediction output linear layer
                                             out_features=self.num_output_classes,
                                             bias=self.use_bias)
        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward prop data through the network and return the preds
        :param x: Input batch x a batch of shape batch number of samples, each of any dimensionality.
        :return: preds of shape (b, num_classes)
        """
        out = x
        out = out.view(out.shape[0], -1)
                         # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            out = self.layer_dict['fcc_{}'.format(i)](out)  # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()

        self.logits_linear_layer.reset_parameters()


class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_shape, dim_reduction_type, num_output_classes, num_filters, num_layers,kernel_size,args,dropout, use_bias=False):
        """
        Initializes a convolutional network module object.
        :param input_shape: The shape of the inputs going in to the network.
        :param dim_reduction_type: The type of dimensionality reduction to apply after each convolutional stage, should be one of ['max_pooling', 'avg_pooling', 'strided_convolution', 'dilated_convolution']
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every conv layer, except dim reduction stages, where those are automatically infered.
        :param num_layers: Number of conv layers (excluding dim reduction stages)
        :param use_bias: Whether our convolutions will use a bias.
        """
        super(ConvolutionalNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dim_reduction_type = dim_reduction_type
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.dropout = dropout
        self.build_module()
        self.stack = args.stack
        print(input_shape)
        print(type(input_shape))
        if self.stack:
           b,c,h,w = self.input_shape
           b = 2*b
           self.input_shape = (b,c,h,w)

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        print("Building basic block of ConvolutionalNetwork using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))  # create dummy inputs to be used to infer shapes of layers

        out = x
                
        
        
        
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        for i in range(self.num_layers):  # for number of layers times
            
            if i == 0:
                 self.layer_dict['batch_{}'.format(i)] = nn.BatchNorm2d(num_features=1)
            if i != 0:
                self.layer_dict['batch_{}'.format(i)] = nn.BatchNorm2d(num_features=self.num_filters[i-1])
            
            
            self.layer_dict['batch_after_conv{}'.format(i)]=nn.BatchNorm2d(num_features=self.num_filters[i])            
            self.layer_dict['conv_{}'.format(i)] = nn.Conv2d(in_channels=out.shape[1],
                                                             # add a conv layer in the module dict
                                                             kernel_size=self.kernel_size,
                                                             out_channels=self.num_filters[i], padding=1,
                                                             bias=self.use_bias)
            
           # self.layer_dict['batch_after_dim'.format(i)]=nn.BatchNorm(num_features=self.num_filters[i])
            # https://arxiv.org/abs/1603.05027
           # if i != 0:
            out = self.layer_dict['batch_{}'.format(i)](out)
            out = F.relu(out)
            out = self.layer_dict['conv_{}'.format(i)](out)  # use layer on inputs to get an output
            out = self.layer_dict['batch_after_conv{}'.format(i)](out) 
            out = F.relu(out)  # apply relu
            
            if self.dim_reduction_type == 'strided_convolution':  # if dim reduction is strided conv, then add a strided conv
                self.layer_dict['dim_reduction_strided_conv_{}'.format(i)] = nn.Conv2d(in_channels=out.shape[1],
                                                                                       kernel_size=3,
                                                                                       out_channels=out.shape[1],
                                                                                       padding=1,
                                                                                       bias=self.use_bias, stride=2,
                                                                                       dilation=1)
                
               # self.layer_dict['batch_{}'.format(i)] = nn.BatchNorm2d(num_features=self.num_filters[i])


                out = self.layer_dict['dim_reduction_strided_conv_{}'.format(i)](
                    out)  # use strided conv to get an output
            #    out = F.relu(out)  # apply relu to the output
            #    out = self.layer_dict['batch_{}'.format(i)](out)
            elif self.dim_reduction_type == 'dilated_convolution':  # if dim reduction is dilated conv, then add a dilated conv, using an arbitrary dilation rate of i + 2 (so it gets smaller as we go, you can choose other dilation rates should you wish to do it.)
                self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)] = nn.Conv2d(in_channels=out.shape[1],
                                                                                       kernel_size=3,
                                                                                       out_channels=out.shape[1],
                                                                                       padding=1,
                                                                                       bias=self.use_bias, stride=1,
                                                                                       dilation=i + 2)
                out = self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)](
                    out)  # run dilated conv on input to get output
             #   out = F.relu(out)  # apply relu on output

            elif self.dim_reduction_type == 'max_pooling':
                self.layer_dict['dim_reduction_max_pool_{}'.format(i)] = nn.MaxPool2d((2,2), padding=1)
                out = self.layer_dict['dim_reduction_max_pool_{}'.format(i)](out)
            elif self.dim_reduction_type == 'avg_pooling':
                self.layer_dict['dim_reduction_avg_pool_{}'.format(i)] = nn.AvgPool2d(2, padding=1)
                out = self.layer_dict['dim_reduction_avg_pool_{}'.format(i)](out)
                
            self.layer_dict['droput_{}'.format(i)] = nn.Dropout2d(self.dropout)
            out = self.layer_dict['droput_{}'.format(i)](out)

            print(out.shape)
        if out.shape[-1] != 2:
            out = F.adaptive_avg_pool2d(out, 2)  # apply adaptive pooling to make sure output of conv layers is always (2, 2) spacially (helps with comparisons).
        print('shape before final linear layer', out.shape)
        out = out.view(out.shape[0], -1)
        self.layer_dict['dropout_linnear'] = nn.Dropout(self.dropout)
        self.logit_linear_layer = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.num_output_classes,
                                            bias=self.use_bias)
        out = self.layer_dict['dropout_linnear'](out)
        out = self.logit_linear_layer(out)  # apply linear layer on flattened inputs
        #print("Block is built, output volume is", out.shape)
        return out

    def forward_train(self, x,epoch_number = -1):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = x
        for i in range(self.num_layers):  # for number of layers
           
           
            
            out = self.layer_dict['batch_{}'.format(i)](out)
           
            out = F.relu(out)
            out = self.layer_dict['conv_{}'.format(i)](out)  # pass through conv layer indexed at i
            #print("memory",torch.cuda.memory_allocated())
            out = self.layer_dict['batch_after_conv{}'.format(i)](out) 
            out = F.relu(out)  # pass conv outputs through ReLU
            if self.dim_reduction_type == 'strided_convolution':  # if strided convolution dim reduction then
                out = self.layer_dict['dim_reduction_strided_conv_{}'.format(i)](out)  # pass previous outputs through a strided convolution indexed i
                out = F.relu(out)  # pass strided conv outputs through ReLU

            elif self.dim_reduction_type == 'dilated_convolution':
                out = self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)](out)
                out = F.relu(out)

            elif self.dim_reduction_type == 'max_pooling':
                out = self.layer_dict['dim_reduction_max_pool_{}'.format(i)](out)

            elif self.dim_reduction_type == 'avg_pooling':
                out = self.layer_dict['dim_reduction_avg_pool_{}'.format(i)](out)
                
            
            out = self.layer_dict['droput_{}'.format(i)](out)

        if out.shape[-1] != 2:
            out = F.adaptive_avg_pool2d(out, 2)
        out = out.view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
        out = self.layer_dict['dropout_linnear'](out)
        out = self.logit_linear_layer(out)  # pass through a linear layer to get logits/preds
        return out
    
    def forward(self, x,epoch_number = -1):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = x
        for i in range(self.num_layers):  # for number of layers
           # if i != 0:
            out = self.layer_dict['batch_{}'.format(i)](out)
            out = F.relu(out)
            out = self.layer_dict['conv_{}'.format(i)](out)  # pass through conv layer indexed at i
            #print("memory",torch.cuda.memory_allocated())
            out = self.layer_dict['batch_after_conv{}'.format(i)](out) 
            out = F.relu(out)  # pass conv outputs through ReLU
            if self.dim_reduction_type == 'strided_convolution':  # if strided convolution dim reduction then
                out = self.layer_dict['dim_reduction_strided_conv_{}'.format(i)](
                    out)  # pass previous outputs through a strided convolution indexed i
                out = F.relu(out)  # pass strided conv outputs through ReLU

            elif self.dim_reduction_type == 'dilated_convolution':
                out = self.layer_dict['dim_reduction_dilated_conv_{}'.format(i)](out)
                out = F.relu(out)

            elif self.dim_reduction_type == 'max_pooling':
                out = self.layer_dict['dim_reduction_max_pool_{}'.format(i)](out)

            elif self.dim_reduction_type == 'avg_pooling':
                out = self.layer_dict['dim_reduction_avg_pool_{}'.format(i)](out)
                

        if out.shape[-1] != 2:
            out = F.adaptive_avg_pool2d(out, 2)
        out = out.view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
        out = self.logit_linear_layer(out)  # pass through a linear layer to get logits/preds
        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

        self.logit_linear_layer.reset_parameters()
        
        
        
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        # build the network
        self.build_module()
        
    def build_module(self):
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.logits_linear_layer = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)
    
    
    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        lstm.reset_parameters()
        self.logits_linear_layer.reset_parameters()
