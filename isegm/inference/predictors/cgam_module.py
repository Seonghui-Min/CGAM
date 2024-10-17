import torch
import torch.nn as nn
import torch.nn.functional as F


class CGAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = None
        self.convf = None
        self.convs = None
        self.convm = None
    
    def initialize(self, image_shape, feature_shape):
        c, self.hf, self.wf = feature_shape[1], feature_shape[2], feature_shape[3]
        hi = image_shape[2]
        self.maxpool = nn.MaxPool2d(round(hi/self.hf))
        self.convf = nn.Conv2d(c, c//2, kernel_size=1)
        self.convs = nn.Conv2d(2, c//2, kernel_size=1)
        
        self.convm = nn.Sequential(
            nn.Conv2d(c//2, c, kernel_size=1),
            nn.InstanceNorm2d(c, affine=True)
        )
        nn.init.constant_(self.convm[1].weight, 0)
        nn.init.constant_(self.convm[1].bias, 1)
        
        self.zero_tensor = torch.zeros(1,2,400,400).to('cuda')
        
    def forward(self, x, click_maps):
        gate_sig = self.maxpool(click_maps)
        
        diffY = x.size()[2] - gate_sig.size()[2]
        diffX = x.size()[3] - gate_sig.size()[3]

        gate_sig = F.pad(gate_sig, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        
        s = self.convs(gate_sig)
        f = self.convf(x)
        
        m = f + s
        m = F.relu(m)
        
        m = self.convm(m)
        
        att = m.squeeze()
        att = torch.mean(att, 0)
        
        return x * m, att