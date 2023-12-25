import torch
import torch.nn as nn
import torch.nn.functional as F

class SEGC(nn.Module):
    def __init__(self,dims):
        super(SEGC,self).__init__()
        self.layer1 = nn.Linear(dims[0],dims[1])
        self.layer2 = nn.Linear(dims[0],dims[1])
        self.layer_increase = nn.Linear(dims[0],dims[1])
        self.layer_decrease = nn.Linear(dims[0],dims[1])
        
    def forward(self,x,x_increase,x_decrease,is_train=True,sigma = 0.01):
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        out_increase = self.layer_increase(x_increase)
        out_decrease = self.layer_decrease(x_decrease)
        
        out1 = F.normalize(out1,dim=1,p=2)
        if is_train:
            out2 = F.normalize(out2, dim=1, p=2) + torch.normal(0, torch.ones_like(out2) * sigma).cuda()
        else:
            out2 = F.normalize(out2, dim=1, p=2)
        out_increase = F.normalize(out_increase,dim=1,p=2)
        out_decrease = F.normalize(out_decrease,dim=1,p=2)
        return out1,out2,out_increase,out_decrease