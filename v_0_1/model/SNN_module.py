"""
-*- coding: utf-8 -*-
__author__:Steve Zhang
2024/5/11 17:08
"""
import torch
import torch.nn as nn
import numpy as np

class Triangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma=1.0):
        out = input.ge(0.0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIFCell(nn.Module):
    """
        leaky-integrate-and-fire neuron dynamic
    """
    def __init__(self, tau_m=0.8, thresh=1.0):
        super(LIFCell, self).__init__()
        self.decay = torch.tensor(tau_m).float()
        self.thresh = thresh
        self.spike_fn = Triangle.apply

    def forward(self, vmem, x):
        vmem = self.decay * vmem + x
        spike = Triangle.apply(vmem - self.thresh)
        vmem -= self.thresh * spike

        return vmem, spike


class TCLIFCell(nn.Module):
    """
        Two-Compartment leaky-integrate-and-fire neuron model
    """
    def __init__(self, tau_m=-1.0, gamma=0.5, thresh=1.0):
        super(TCLIFCell, self).__init__()
        decay = torch.full([1, 2], tau_m, dtype=torch.float)
        self.decay = torch.nn.Parameter(decay)
        self.thresh = thresh
        self.gamma = gamma
        self.spike_fn = Triangle.apply

    def forward(self, d_vmem, s_vmem, x):
        d_vmem = d_vmem - torch.sigmoid(self.decay[0][0]) * s_vmem + x
        s_vmem = s_vmem + torch.sigmoid(self.decay[0][1]) * d_vmem
        spike = Triangle.apply(s_vmem - self.thresh)
        d_vmem = d_vmem - self.gamma * spike
        s_vmem = s_vmem - self.thresh * spike
        # print(torch.sum(spike) / spike.numel())

        return d_vmem, s_vmem, spike


class SNNLayer(nn.Module):
    def __init__(self, cell=LIFCell, if_hid=1, **cell_args):
        super(SNNLayer, self).__init__()
        self.cell = cell(**cell_args)
        self.if_hid = if_hid

    def forward(self, x):
        vmem = torch.zeros_like(x[0].data)
        # d_vmem = torch.zeros_like(x[0].data)
        spikes = []
        vmems = []
        for step in range(x.size(0)):
            # print(step)
            current = x[step]
            vmem, spike = self.cell(vmem, current)
            spikes.append(spike)
            vmems.append(vmem)
        return torch.stack(spikes).mean(0) if self.if_hid else torch.stack(vmems)


class SNNRecLayer(nn.Module):
    def __init__(self, cell=LIFCell, hid_size=64, **cell_args):
        super(SNNRecLayer, self).__init__()
        self.hid_size = hid_size
        self.linear = nn.Linear(hid_size, hid_size)
        self.cell = cell(**cell_args)

    def forward(self, x):
        vmem = torch.zeros_like(x[0].data)
        # d_vmem = torch.zeros_like(x[0].data)
        spike = torch.zeros_like(x[0].data)
        spikes = []
        for step in range(x.size(0)):
            current = x[step] + self.linear(spike)
            vmem, spike = self.cell(vmem, current)
            spikes.append(spike)
        return torch.stack(spikes).mean(0)


if __name__ == '__main__':
    layer = SNNLayer(cell=LIFCell, if_hid=0, tau_m=0.5, thresh=1.0)
    # ann_layer = Frame1_10s()
    x = torch.rand((412, 1, 64, 1280))
    x = x.squeeze()
    x = x.permute(2, 0, 1).contiguous()
    Linear1 = nn.Linear(64, 128)
    Linear2 = nn.Linear(128, 5)
    x = Linear1(x)
    x = layer(x)
    # x = Linear2(x)

    # x = x.permute(2, 0, 1).contiguous()
    # res = layer(x)

    print(x.shape)