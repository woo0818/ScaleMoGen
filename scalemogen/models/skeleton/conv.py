"""ScaleMoGen skeleton convolution layers.

Code provenance: adapted from the SALAD skeleton graph convolution layers.
Source repository: https://github.com/seokhyeonhong/salad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name):
    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "gelu":
        return nn.GELU()
    elif name.lower() == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")


def get_norm(name, dim):
    if name.lower() == "layer":
        return nn.LayerNorm(dim)
    elif name.lower() == "batch":
        return nn.BatchNorm1d(dim)
    elif name.lower() == "group":
        return nn.GroupNorm(32, dim)
    elif name.lower() == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization function: {name}")


class GraphConv(nn.Module):
    """
    Graph Convolution.
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.linear2 = nn.Linear(in_channels, out_channels, bias=bias)
    
    
    def forward(self, x, adj_matrix):
        """
        x: [B, T, J, D]
        adj_matrix: [J, J]

        return: linear1(x) + sum_{j \in N(i)} linear2(x_j)
        """

        h1 = self.linear1(x) # [B, T, J, D]
        h2 = torch.matmul(adj_matrix, self.linear2(x)) / (adj_matrix.sum(dim=-1, keepdim=True) + 1e-6) # [B, T, J, D]

        out = h1 + h2

        return out


class STConv(nn.Module):
    """
    Skeleto-Temporal Convolution.
    """
    def __init__(
        self,
        edges,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
    ):
        assert kernel_size % 2 == 1, f"kernel_size should be odd number, but got {kernel_size}."

        super(STConv, self).__init__()

        # grpah convolution
        self.graph_conv = GraphConv(in_channels, out_channels, bias=bias)
        self.adj_matrix = nn.Parameter(self._get_adj_matrix(edges), requires_grad=False) # symmetric matrix

        # temporal convolution
        self.temp_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
        self._use_cudnn_temporal_conv = True
    
    def _get_adj_matrix(self, edges, add_self_loop=True):
        max_idx = -1
        for i, j in edges:
            max_idx = max(max_idx, i, j)

        adj_matrix = torch.zeros(max_idx + 1, max_idx + 1)
        for i, j in edges:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        
        if add_self_loop:
            for i in range(max_idx + 1):
                adj_matrix[i, i] = 1
        
        return adj_matrix

    def forward(self, x):
        B, T, J, D = x.size()

        # graph conv
        graph_out = self.graph_conv.forward(x, self.adj_matrix)

        # temporal conv
        temp_in = x.permute(0, 2, 3, 1).reshape(B * J, D, T)
        if self._use_cudnn_temporal_conv:
            try:
                temp_out = self.temp_conv(temp_in)
            except RuntimeError as exc:
                if "unable to find an engine" not in str(exc):
                    raise
                self._use_cudnn_temporal_conv = False
                temp_out = self._fallback_temporal_conv(temp_in)
        else:
            temp_out = self._fallback_temporal_conv(temp_in)
        temp_out = temp_out.reshape(B, J, -1, T).permute(0, 3, 1, 2)
        
        out = graph_out + temp_out

        return out

    def _fallback_temporal_conv(self, temp_in):
        """Run temporal Conv1d without cuDNN when a CUDA conv engine is unavailable."""
        with torch.backends.cudnn.flags(enabled=False):
            return F.conv1d(
                temp_in,
                self.temp_conv.weight,
                self.temp_conv.bias,
                stride=self.temp_conv.stride,
                padding=self.temp_conv.padding,
                dilation=self.temp_conv.dilation,
                groups=self.temp_conv.groups,
            )
    

class ResSTConv(nn.Module):
    """
    Residual Skeleto-Temporal Convolution.
    """
    def __init__(
        self,
        edges,
        dim_channels,
        kernel_size,
        bias=True,
        activation="gelu",
        norm="none",
        dropout=0.1,
    ):
        super(ResSTConv, self).__init__()
        self.layers = nn.Sequential(
            get_norm(norm, dim_channels),
            get_activation(activation),
            STConv(edges, dim_channels, dim_channels, kernel_size, bias=bias),
            get_norm(norm, dim_channels),
            get_activation(activation),
            STConv(edges, dim_channels, dim_channels, kernel_size, bias=bias),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.layers(x)
