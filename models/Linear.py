import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fairseq.modules.moving_average_gated_attention import MovingAverageGatedAttention

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        self.mov_avg = MovingAverageGatedAttention(
            embed_dim=1,
            zdim=4,
            hdim=8,
            ndim=8,
            rel_pos_bias = 'rotary',
            attention_activation = 'element'
            )
    

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        
        out,_ = self.mov_avg(x.permute(1,0,2)) # out: [Time,Batch,Channel]
        x = self.Linear(out.permute(1,2,0)).permute(0,2,1)
        return x # [Batch, Output length, Channel]