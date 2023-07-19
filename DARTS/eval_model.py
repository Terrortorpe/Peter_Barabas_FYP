import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from typing import Any

class Genotype:
    normal: Any
    normal_concat: Any
    reduce: Any
    reduce_concat: Any

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(x.device))
        x.div_(keep_prob)
        x.mul_(mask)
    return x

class Cell(nn.Module):
  def __init__(self, genotype, C_prev_prev, C_prev, C, is_reduction, reduction_prev):
    super(Cell, self).__init__()

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)

    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if is_reduction:
      operation_names, indices = zip(*genotype.reduce)
      self.concatenation = genotype.reduce_concat
    else:
      operation_names, indices = zip(*genotype.normal)
      self.concatenation = genotype.normal_concat

    assert len(operation_names) == len(indices)

    self._steps = len(operation_names) // 2
    self.multiplier = len(self.concatenation)

    self._ops = nn.ModuleList([OPS[name](C, 2 if is_reduction and index < 2 else 1, True) for name, index in zip(operation_names, indices)])
    self._indices = indices

  def forward(self, s0, s1, drop_probability):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1, h2 = states[self._indices[2*i]], states[self._indices[2*i+1]]
      op1, op2 = self._ops[2*i], self._ops[2*i+1]

      h1, h2 = op1(h1), op2(h2)

      if self.training and drop_probability > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_probability)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_probability)
      
      states.append(h1 + h2)

    return torch.cat([states[i] for i in self.concatenation], dim=1)

class SampledNetwork(nn.Module):
  def __init__(self, C, num_classes, layers, genotype):
    super(SampledNetwork, self).__init__()

    self._layers = layers
    stem_multiplier = 3
    C_curr = stem_multiplier * C

    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev = C_curr, C_curr

    self.cells = nn.ModuleList()
    reduction_prev = False

    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells.append(cell)

      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input_data):
    s0 = s1 = self.stem(input_data)

    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits
