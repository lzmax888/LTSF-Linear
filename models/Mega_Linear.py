import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from mega_pytorch import MegaLayer

from fairseq.modules.moving_average_gated_attention import MovingAverageGatedAttention

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.batch  = configs.batch_size
       

        # Decompsition Kernel Size
        kernel_size = configs.moving_avg
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in
        
        # self.magelayer_seasonal = MegaLayer(
        #     dim = 1,                   # model dimensions
        #     ema_heads = 4,              # number of EMA heads
        #     attn_dim_qk = 32,            # dimension of queries / keys in attention
        #     attn_dim_value = 32,        # dimension of values in attention
        #     laplacian_attn_fn = False,   # whether to use softmax (false) or laplacian attention activation fn (true)
        # )
        
        # self.magelayer_trend = MegaLayer(
        #     dim = 1,                   # model dimensions
        #     ema_heads = 16,              # number of EMA heads
        #     attn_dim_qk = 64,            # dimension of queries / keys in attention
        #     attn_dim_value = 256,        # dimension of values in attention
        #     laplacian_attn_fn = False,   # whether to use softmax (false) or laplacian attention activation fn (true)
        # )
        
        self.mov_avg_seasonal = MovingAverageGatedAttention(
            embed_dim=1,
            zdim=8,
            hdim=8,
            ndim=8,
            )
        
        self.mov_avg_trend = MovingAverageGatedAttention(
            embed_dim=1,
            zdim=8,
            hdim=8,
            ndim=8,
            )

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

            #self.Linear_Seasonal_RNN = nn.GRU(input_size = 1,hidden_size = 1, batch_first=True,num_layers=1)
            #self.Linear_Trend_RNN = nn.GRU(input_size = 1,hidden_size = 1, batch_first=True,num_layers=1)

            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        #seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            #seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
            #print(seasonal_init.shape)
            seasonal_output,_ = self.mov_avg_seasonal(seasonal_init.permute(1,0,2)) # [T,B,C]
            trend_output,_ = self.mov_avg_trend(trend_init.permute(1,0,2))

            seasonal_output = self.Linear_Seasonal(seasonal_output.permute(1,2,0)) # [B,C,T]
            trend_output = self.Linear_Trend(trend_output.permute(1,2,0))

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
        
        #return x
