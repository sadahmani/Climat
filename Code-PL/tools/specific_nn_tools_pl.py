#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specific NN Tools for PL

Contains classes or functions:
    
    class CustomDataset(Dataset)
    
    class Net(nn.Module)

    class NetORIG(nn.Module)

    # TORCH_SUMMARIZE:
    # Summarizes torch model by showing trainable parameters and weights.
    #
    # Usage:
        tmpstr = torch_summarize (model, show_weights=True, show_parameters=True)

    # GET_DEVICE_AUTO:
    #
    # Usage:
        device = def get_device_auto(device=None, ngpu=1,verbose=False)


Created on Mon Aug 22 11:30:19 2022

@author: carlos.mejia@locean.ipsl.fr
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self,ghg,aer,nat,historical):
        self.ghg = ghg
        self.aer = aer
        self.nat = nat
        self.historical = historical

    def __len__(self):
        return len(self.ghg)

    def __getitem__(self, item):
        import torch
        
        X = torch.stack((self.ghg[item,:], self.aer[item,:], self.nat[item,:]),dim=0)
        Y = self.historical[item,:]

        return X,Y


class CustomDatasetInv(Dataset):
    def __init__(self,ghg,aer,nat):
        self.ghg = ghg
        self.aer = aer
        self.nat = nat

    def __len__(self):
        return len(self.ghg)

    def __getitem__(self, item):
        #X = torch.stack((self.ghg[item,:], self.aer[item,:], self.nat[item,:]),dim=0).float()
        X = torch.stack((self.ghg[item,:], self.aer[item,:], self.nat[item,:]),dim=0)
        return X


class Net(nn.Module):
    """
        Alows between initialisation and forward methods for CONV1D networks.
        Accept from 2 to 6 hidden CNN layers with size given in the parameter
        conv_layers_size of the __init__() constructor method. 
    """
    def __init__(self, size_channel, conv_layers_size, bias=True, input_chanel_size=3, output_chanel_size=1, type=0, verbose=False):
        import numpy as np
        
        super(Net, self).__init__()
        self.type = type
        self.tanh = nn.Tanh()
        
        if np.isscalar(conv_layers_size):
            conv_layers_size = [conv_layers_size]
        if np.isscalar(size_channel) :
            size_channel = [size_channel]
        
        self.n_conv = np.max((len(conv_layers_size),len(size_channel)))
        
        if len(conv_layers_size) == 1 and self.n_conv > 1 :
            conv_layers_size = conv_layers_size * self.n_conv
        if len(size_channel) == 1 and self.n_conv > 1 :
            size_channel = size_channel * self.n_conv
        
        if verbose:
            print(f"Net init ... conv_layers_size: {conv_layers_size}")
            print(f"Net init ... n_conv: {self.n_conv}")
            print(f"Net init ... conv_layers_size: {conv_layers_size}")
            print(f"Net init ... size_channel: {size_channel}")

        if self.n_conv == 2 :
            iconv  = 0; self.conv1 = nn.Conv1d(input_chanel_size, size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.convOut = nn.Conv1d(size_channel[-1], output_chanel_size, kernel_size=conv_layers_size[-1], bias=bias)

        elif self.n_conv == 3 :
            iconv  = 0; self.conv1 = nn.Conv1d(input_chanel_size, size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.conv2 = nn.Conv1d(size_channel[iconv-1], size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.convOut = nn.Conv1d(size_channel[-1], output_chanel_size, kernel_size=conv_layers_size[-1], bias=bias)

        elif self.n_conv == 4 :
            iconv  = 0; self.conv1 = nn.Conv1d(input_chanel_size, size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.conv2 = nn.Conv1d(size_channel[iconv-1], size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.conv3 = nn.Conv1d(size_channel[iconv-1], size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.convOut = nn.Conv1d(size_channel[-1], output_chanel_size, kernel_size=conv_layers_size[-1], bias=bias)

        elif self.n_conv == 5 :
            iconv  = 0; self.conv1 = nn.Conv1d(input_chanel_size, size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.conv2 = nn.Conv1d(size_channel[iconv-1], size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.conv3 = nn.Conv1d(size_channel[iconv-1], size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.conv4 = nn.Conv1d(size_channel[iconv-1], size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.convOut = nn.Conv1d(size_channel[-1], output_chanel_size, kernel_size=conv_layers_size[-1], bias=bias)
            
        elif self.n_conv == 6 :
            iconv  = 0; self.conv1 = nn.Conv1d(input_chanel_size, size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.conv2 = nn.Conv1d(size_channel[iconv-1], size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.conv3 = nn.Conv1d(size_channel[iconv-1], size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.conv4 = nn.Conv1d(size_channel[iconv-1], size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.conv5 = nn.Conv1d(size_channel[iconv-1], size_channel[iconv], kernel_size=conv_layers_size[iconv], bias=bias)
            iconv += 1; self.convOut = nn.Conv1d(size_channel[-1], output_chanel_size, kernel_size=conv_layers_size[-1], bias=bias)
       
        else:
            raise f"Net __init_ invalid architecture (number of convolution layers, 'n_conv={self.n_conv}'). Min 2, max 6 (for instance)"

    def forward(self, X):
        #L = X.shape[2]

        x = self.conv1(X)
        x = self.tanh(x)
        if self.n_conv == 2 :
            x = self.convOut(x)[:,0,:]
        else :
            x = self.conv2(x)
            x = self.tanh(x)
            if self.n_conv == 3 :
                x = self.convOut(x)[:,0,:]
            else :
                x = self.conv3(x)
                x = self.tanh(x)
                if self.n_conv == 4 :
                    x = self.convOut(x)[:,0,:]
                else :
                    x = self.conv4(x)
                    x = self.tanh(x)
                    if self.n_conv == 5 :
                        x = self.convOut(x)[:,0,:]
                    else :
                        x = self.conv5(x)
                        x = self.tanh(x)
                        if self.n_conv == 6 :
                            x = self.convOut(x)[:,0,:]
                        else :
                            raise f"forward ... invalid number of convolution layers 'n_conv={self.n_conv}'. Min 2, max 6 (for instance)"

        return x


class NetORIG(nn.Module):

    def __init__(self,size_channel,k1,k2,k3,bias,type=0):
        super(Net, self).__init__()
        self.type = type
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv1d(3,size_channel,kernel_size=k1,bias=bias)
        self.conv2 = nn.Conv1d(size_channel, size_channel, kernel_size=k2,bias=bias)
        self.conv3 = nn.Conv1d(size_channel, 1, kernel_size=k3, bias=bias)

    def forward(self, X):
        #L = X.shape[2]

        x = self.conv1(X)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.tanh(x)
        #x = x.float()
        x = self.conv3(x)[:,0,:]

        return x


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    from torch.nn.modules.module import _addindent
    import numpy as np

    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   
    
    tmpstr = tmpstr + ')'
    
    return tmpstr


def get_device_auto(device=None, ngpu=1,verbose=False):

    if device is None or device.lower() in ['gpu','mps'] :
        if torch.cuda.is_available() :
            #device = f"cuda:{ngpu}"
            device = torch.device(f"cuda:{ngpu}")
            torch.cuda.set_device(device)
            if verbose:
                print('Torch cuda device=',torch.cuda.current_device(),
                      '\nProperties .......',torch.cuda.get_device_properties(torch.cuda.current_device()))
        
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print(device)
            #device = "mps"
            torch.device(device)
            if verbose:
                print('Torch device=',device)
        else:
            print(f" ** Cannot set device to {device}. Setting to None **")
            device=None
    
    if device is None:
        device = "cpu"
    
    if verbose:
        print('Currently used device is :', device)
    
    return device
