import torch
from operations import *
from typing import Any

class Genotype:
    normal: Any
    normal_concat: Any
    reduce: Any
    reduce_concat: Any
    
BASE_OP_NAMES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

class MixedOp(torch.nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = torch.nn.ModuleList()
    for primitive in BASE_OP_NAMES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = torch.nn.Sequential(op, torch.nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    result = 0
    for weight, operation in zip(weights, self._ops):
        result += weight * operation(x)
    return result


class Cell(torch.nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self.nodes = steps
    self._multiplier = multiplier

    self._ops = torch.nn.ModuleList()
    self._bns = torch.nn.ModuleList()
    for i in range(self.nodes):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self.nodes):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Supernet(torch.nn.Module):

  def __init__(self, C, classes, loss_func, num_cells=8, nodes=4, multiplier=4):
    super(Supernet, self).__init__()
    
    self.loss_func = loss_func
    self._multiplier = multiplier
    self._C = C 
    self.num_cells = num_cells
    self.classes = classes
    self.nodes = nodes

    C_current = 3 * C
    
    self.network_stem = torch.nn.Sequential(
      torch.nn.Conv2d(3, 3*C, 3, padding=1, bias=False),
      torch.nn.BatchNorm2d(C_current)
    )
    self.cells = torch.nn.ModuleList()
    reduction_prev = False
 
    C_pprev, C_prev, C_current = C_current, C_current, C
    self.cell_modules = torch.nn.ModuleList()
    reduction_checkpoints = [num_cells//3, 2*num_cells//3]
    for idx in range(num_cells):
        reduction_now = idx in reduction_checkpoints
        if reduction_now:
            C_current *= 2
        cell_module = Cell(nodes, multiplier, C_pprev, C_prev, C_current, reduction_now, idx > 0 and idx - 1 in reduction_checkpoints)
        self.cell_modules.append(cell_module)
        C_pprev, C_prev = C_prev, multiplier * C_current

    self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
    self.output_layer = torch.nn.Linear(C_prev, self.classes)

    num_connections = sum(list(range(2, 2+self.nodes)))
    num_ops = len(BASE_OP_NAMES)
    self.normal_parameters = torch.randn(num_connections, num_ops, device='cuda') * 1e-2
    self.normal_parameters.requires_grad = True

    self.reduction_parameters = torch.randn(num_connections, num_ops, device='cuda') * 1e-2
    self.reduction_parameters.requires_grad = True

    self.architecture_parameters = [
        self.normal_parameters,
        self.reduction_parameters,
    ]
    
    #-----------------------------
    # asd = torch.full((k, num_ops), 0.1)
    # for i in range(shape(asd)[0]):
      # asd[i][3] = 2
    # self.normal_parameters = Variable(asd.cuda(), requires_grad=True)
    #-----------------------------

  def forward(self, input_data):
    s0 = s1 = self.network_stem(input_data)
    
    weights_calc = lambda parameters: torch.nn.functional.softmax(parameters, dim=-1)

    for idx, cell in enumerate(self.cell_modules):
        weights = weights_calc(self.reduction_parameters if cell.reduction else self.normal_parameters)
        s0, s1 = s1, cell(s0, s1, weights)
      
    out = self.global_avg_pool(s1).view(s1.size(0), -1)
    logits = self.output_layer(out)
    return logits 

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self.nodes):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != BASE_OP_NAMES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != BASE_OP_NAMES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((BASE_OP_NAMES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(torch.nn.functional.softmax(self.normal_parameters, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(torch.nn.functional.softmax(self.reduction_parameters, dim=-1).data.cpu().numpy())

    concat = range(2+self.nodes-self._multiplier, self.nodes+2)
    genotype = Genotype()
    genotype.normal=gene_normal,
    genotype.normal_concat=concat,
    genotype.reduce=gene_reduce,
    genotype.reduce_concat=concat
    return genotype