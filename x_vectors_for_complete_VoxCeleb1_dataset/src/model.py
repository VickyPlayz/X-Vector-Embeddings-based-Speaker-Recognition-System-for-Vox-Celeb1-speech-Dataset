import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(TDNNLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))

class StatsPooling(nn.Module):
    def __init__(self):
        super(StatsPooling, self).__init__()

    def forward(self, x):
        """
        x: [Batch, Dim, Time]
        Returns: [Batch, 2*Dim] (mean + std)
        """
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)

class XVectorNet(nn.Module):
    def __init__(self, input_dim=24, num_speakers=1251):
        super(XVectorNet, self).__init__()
        
        # Frame-level (TDNN) layers
        # Ctx {-2,-1,0,1,2} -> Kernel 5, Dilation 1
        self.tdnn1 = TDNNLayer(input_dim, 512, kernel_size=5, dilation=1)
        # Ctx {-2,0,2} -> Kernel 3, Dilation 2
        self.tdnn2 = TDNNLayer(512, 512, kernel_size=3, dilation=2)
        # Ctx {-3,0,3} -> Kernel 3, Dilation 3
        self.tdnn3 = TDNNLayer(512, 512, kernel_size=3, dilation=3)
        # Ctx {0} -> Kernel 1
        self.tdnn4 = TDNNLayer(512, 512, kernel_size=1, dilation=1)
        self.tdnn5 = TDNNLayer(512, 1500, kernel_size=1, dilation=1)
        
        # Pooling
        self.pool = StatsPooling()
        
        # Segment-level layers
        self.seg1 = nn.Linear(3000, 512)
        self.seg2 = nn.Linear(512, 512)
        
        # Output layer (Softmax)
        self.output = nn.Linear(512, num_speakers)
        
    def forward(self, x, return_embedding=False):
        """
        x: [Batch, Input_Dim, Time]
        """
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        
        x = self.pool(x) # [Batch, 3000]
        
        x = F.relu(self.seg1(x))
        # Embedding is usually taken from the affine part of this layer (before ReLU?) 
        # Or often the second segment layer. 
        # Standard x-vector uses the output of the first or second segment layer.
        # Let's use the output of seg2 as the embedding (before classification).
        
        x_embedding = self.seg2(x) # [Batch, 512]
        
        if return_embedding:
            return x_embedding
            
        x = F.relu(x_embedding)
        logits = self.output(x)
        return logits
