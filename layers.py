import torch
from torch import nn
import numpy as np
import math

def t2v(tau, f, out_features, w, b, w0, b0):
    # k-1 periodic features
    v1 = f(torch.matmul(tau, w) + b)
    # One Non-periodic feature
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 2)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)