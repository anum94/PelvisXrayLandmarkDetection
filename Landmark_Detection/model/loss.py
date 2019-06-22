import torch.nn.functional as F
import torch
from torch.nn import BCEWithLogitsLoss, Sigmoid, BCELoss
#from utils import util
import torch.nn.functional as F
from torch import nn



def mse_loss(output, target):  
    return F.mse_loss(output.double(), target.double())

def nll_loss(output, target):
    return F.nll_loss(output, target)


#def smooth_l1_loss(output, target):
#    return util.apply_loss(nn.SmoothL1Loss(reduction='sum'), output, target)


def bce_loss(output, target):
    batch_size = output.size(0)

    criterion = BCELoss()
    batch_size = output.size(0)
    output_flat = output.reshape(batch_size,23,-1)
    target_flat = target.reshape(batch_size,23,-1)
    print(output_flat.size())
    print(target_flat.size())
    return criterion(output_flat, target_flat)



def soft_dice_loss(output, target, epsilon = 1e-6):
    batch_size = output.size(0)
    sig = Sigmoid()    
    output_prob = sig(output)
    #x = torch.ones(1,1)
    #y = torch.zeros(1,1)
    #output_prob = torch.where(output>=1.,x,y)
    output_f = output_prob.reshape(batch_size,-1)
    target_f = target.reshape(batch_size,-1)
    numerator = 2. * (output_f* target_f).sum(0)
    denominator = (output_f*output_f + target_f*target_f).sum(0)
    loss = 1- numerator/(denominator+epsilon)
    print(loss.size())
    
    return loss.mean()

