import os
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import datetime
import torchvision
from eval_model import SampledNetwork
from eval_model import Genotype


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--loader_workers', type=int, default='0', help='number of data loading workers')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

# Logging
#---------------------------------------------------------
logger = logging.getLogger("architecture_evaluation")
logger.setLevel(logging.INFO)

#create a directory for the log file
os.makedirs('logs', exist_ok=True)

curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
files = logging.FileHandler(f'./logs/AS_{curr_time}.log')
files.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

formatting = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
files.setFormatter(formatting)
console.setFormatter(formatting)

logger.addHandler(files)
logger.addHandler(console)
#--------------------------------------------------------- 

def get_training_data():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
        
    train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    
    return train_data

def get_validation_data():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    valid_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
        
    valid_data = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    
    return valid_data


def make_training_loader(training_data):
    train_loader = torch.utils.data.DataLoader( # type: ignore
        training_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.loader_workers)
    return train_loader

def make_validation_loader(validation_data):
    valid_loader = torch.utils.data.DataLoader( # type: ignore
        validation_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.loader_workers)
    return valid_loader

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
    
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res 

CLASSES = 10

def main():
  logger.info("Starting architecture evaluation")
  for arg_name in vars(args):
      arg_value = getattr(args, arg_name)
      logger.info(f"{arg_name}: {arg_value}")

  # Check gpu
  if torch.cuda.is_available():
      logger.info("Using GPU")
      torch.cuda.set_device(args.gpu)
      torch.backends.cudnn.benchmark = True # type: ignore
      torch.backends.cudnn.enabled = True # type: ignore
  else:
      logger.warning("NO GPU DETECTED")
      raise Exception("NO GPU DETECTED")

  np.random.seed(args.seed)

  #Training data
  train_data = get_training_data()
  train_loader = make_training_loader(train_data)

  #Validation data
  valid_data = get_validation_data()
  valid_loader = make_validation_loader(valid_data)
  
  genotype = Genotype()
  
  genotype.normal = [('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)] 
  genotype.normal_concat=[2, 3, 4, 5]
  genotype.reduce = [('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)]
  genotype.reduce_concat=[2, 3, 4, 5]
  
  model = SampledNetwork(args.init_channels, CLASSES, args.layers, genotype)
  model = model.cuda()

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )
  
  loss_func = nn.CrossEntropyLoss().cuda()

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

  for epoch in range(args.epochs):
    logger.info(f'epoch {epoch}')
    
    model.train()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    
    for batch_no, (input, gt) in enumerate(train_loader):
      optimizer.zero_grad()
      
      input = input.to('cuda')
      gt = gt.to('cuda')

      output = model(input)
      loss = loss_func(output, gt)
      
      loss.backward()
      
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()

      top1new, top5new = accuracy(output, gt, topk=(1, 5))
      batch_size = input.size(0)
      top1.update(top1new.item(), batch_size)
      top5.update(top5new.item(), batch_size)
      
      logger.info(f'train {batch_no} {top1.avg} {top5.avg}')

    scheduler.step()

    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    for batch_no, (input, gt) in enumerate(valid_loader):
      input = input.to('cuda')
      gt = gt.to('cuda')

      with torch.no_grad():
        output = model(input)
        loss = loss_func(output, gt)

      top1new, top5new = accuracy(output, gt, topk=(1, 5))
      batch_size = input.size(0)
      top1.update(top1new.item(), batch_size)
      top5.update(top5new.item(), batch_size)
      
      logger.info(f'validation {batch_no} {top1.avg} {top5.avg}')

if __name__ == '__main__':
  main() 