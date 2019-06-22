# -*- coding: utf-8 -*-
"""
Created on Thu May 16 03:38:49 2019

@author: felix
"""

import model.model as module_arch
import model.loss as module_loss
import model.metric as module_metric

from parse_config import ConfigParser


def main(config, resume):
    training_set = Dataset(data_dir+"/train")
    data_loader = DataLoader(training_set, batch_size = batch_size)
    
    validation_set = Dataset(data_dir + "/validation")
    valid_data_loader = DataLoader(validation_set, batch_size=2)
    
    
    model = config.initialize('arch', module_arch)

    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    
    model.eval()