import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import types

def snip_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)

def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)

def get_layer_metric_array(net, metric_fn):
    metric_list = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            metric = metric_fn(layer)
            metric_list.append(metric.view(-1).detach().cpu().numpy())
    return np.concatenate(metric_list)

def snip(net, dataloader, loss_fn):
    net.eval()
    net.to('cuda')

    # Data acquisition
    inputs, targets = next(iter(dataloader))
    inputs = inputs.to('cuda')
    targets = targets.to('cuda')

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    net.zero_grad()
    outputs = net(inputs)[0]
    loss = loss_fn(outputs, targets)
    loss.backward()

    def snip(layer):
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad.data)
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, snip)
    score = np.sum(grads_abs)

    net.train()

    return score