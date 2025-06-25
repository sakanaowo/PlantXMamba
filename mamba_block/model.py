# PlantXMamba/mamba_block/model.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# from mamba_block.backbone import Mamba, MambaConfig
# from mamba_block.head import MambaHead

from backbone import Mamba, MambaConfig
from head import MambaHead


# class MambaModule(nn.Module):
#     def __init__(
#         self, args
#     ):
#         """
#         Args:
#             d_model: Hidden dimension size
#             n_layers: Number of Mamba layers
#             output_size: dimension of embedding space
#             dropout: Dropout rate for head
#         """
#         super().__init__()
#         self.args = args
#         self.d_model = self.args.d_model
#         self.n_layers = self.args.n_layers
#         self.output_size = self.args.output_size
#         self.dropout = self.args.dropout
        
#         # self.d_model = 2048
#         # self.n_layers = 2
#         # self.output_size = 64
#         # self.dropout = 0.1
        
#         config = MambaConfig(d_model=self.d_model, n_layers=self.n_layers)
#         self.backbone = Mamba(config)
#         self.head = MambaHead(d_model=self.d_model, 
#                               output_size=self.output_size, dropout=self.dropout)
        
#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor of shape (batch_size, seq_len, channels) for embeddings
#         """
#         sequence_output = self.backbone(x) # output: (batch_size, seq_len, channels)
        
#         # return F.normalize(sequence_output[:, -1, :])
        
#         output = self.head(sequence_output)        
#         return output # (N, 64)
    

class MambaModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = self.args.d_model
        self.n_layers = self.args.n_layers
        
        config = MambaConfig(d_model=self.d_model, n_layers=self.n_layers, 
                           d_state=self.args.d_state, d_conv=self.args.d_conv, 
                           expand_factor=self.args.expand)
        self.backbone = Mamba(config)
        self.head = MambaHead(d_model=self.d_model, dropout=0.0)
        
    def forward(self, x):
        sequence_output = self.backbone(x)  # (batch_size, seq_len, d_model)
        output = self.head(sequence_output)  # (batch_size, seq_len, d_model)
        return output

# model = MambaModule(None)
# dummy_input = torch.randn(1, 25, 2048)
# output = model(dummy_input)
# print("Output shape:", output.shape) # (1, 25, 2048)