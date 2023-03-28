import torch
import numpy as np
from torch.nn import functional as F
eps=1e-8

class SharpeLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, ):
        super().__init__()
        self.lam = 0.01

    def sharpe_ratio(self, Rp, pred_sz):
        """
        Rp: Portfolio returns over a trading period, [BS,L]
        """
        sharpe_list = torch.zeros(Rp.shape[0])
        sharpe = torch.zeros(Rp.shape[0])
        expected_Rp = []
        variance_Rp = []
        for i in range(len(Rp)):
            expected_Rp.append(torch.mean(Rp[i,:pred_sz[i]]))
            variance_Rp.append(torch.var(Rp[i,:pred_sz[i]]))
            sharpe[i] = expected_Rp[i]/(torch.sqrt(variance_Rp[i] + eps))
            sharpe_list[i] = expected_Rp[i] - self.lam/(torch.sqrt(variance_Rp[i] + eps))
        return sharpe_list, sharpe, torch.tensor(expected_Rp), torch.tensor(variance_Rp)

    def arithmetic_return(self, P_i, P_j):
        """
        P_i: Price tensor on [0,i-1] days, [BS,L,N] (past prices)
        P_j: Price tensor on [1, i] days, [BS,L,N] (future prices)
        """
        arithmetic_r = (P_j-P_i)/(P_i)
        arithmetic_r = torch.nan_to_num(arithmetic_r)
        mask = arithmetic_r <= 2
        arithmetic_r *= mask
        return arithmetic_r

    def portfolio_return(self, weights, arithmetic_return):
        """
        weights: [BS,L,N]
        arithmetic_return: [BS,L,N]
        """
        return torch.sum(weights*arithmetic_return, dim=-1)

    def forward(self, pred_weights, price_list, pred_sz):
        """
        pred_weigths: [BS,L,N]
        price_list: [BS,L+1,N]
        pred_sz: [BS,1]
        """
        #TODO: calculate loss for only those values where attn_mask is True(excluding next and past variable)
        arithmetic_returns = self.arithmetic_return(price_list[:, :-1, :], price_list[:, 1:, :])  ## [BS,L]
        Rp = self.portfolio_return(pred_weights[:,:price_list.shape[1]-1,:], arithmetic_returns) # [BS,L,]
        sharpe_list, sharpe, expected_Rp, variance_Rp = self.sharpe_ratio(Rp, pred_sz) # [BS]
        return -1*torch.mean(sharpe_list), -torch.mean(sharpe), torch.mean(expected_Rp), torch.mean(variance_Rp)