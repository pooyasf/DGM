#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:48:03 2020

@author: Pooya
"""
from libs import *

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation = output.detach()
    return hook


#hooks = {}
#for name, module in net.named_modules():
#    hooks[name] = module.register_forward_hook(get_activation(name))
def register_hook(net):
    net.fc_input.register_forward_hook(get_activation('fc_input'))
    net.linears[0].register_forward_hook(get_activation('linears[0]'))
    net.linears[1].register_forward_hook(get_activation('linears[1]'))
    net.fc_output.register_forward_hook(get_activation('fc_output'))


