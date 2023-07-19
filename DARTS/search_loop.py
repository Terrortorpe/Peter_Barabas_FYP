import os
import numpy as np
import torch
import logging
import argparse
import datetime
import torchvision

from search_model import Supernet


# Arguments
# Copied from original DARTS code
parser = argparse.ArgumentParser("architecture search")
parser.add_argument('--classes', type=int, default=10, help='number of classes')
parser.add_argument('--loader_workers', type=int, default='0', help='number of data loading workers')
parser.add_argument('--data', type=str, default='../data', help='location of the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='training epochs')
parser.add_argument('--channels', type=int, default=16, help='number of channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=6, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.1, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

# Logging
#---------------------------------------------------------
logger = logging.getLogger("architecture_search")
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
        
    dataset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[:split], indices[split:]
    train_data = torch.utils.data.Subset(dataset, train_idx)
    
    return train_data

def get_validation_data():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    valid_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
        
    dataset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[:split], indices[split:]
    valid_data = torch.utils.data.Subset(dataset, valid_idx)
    
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

def flattener(tensor_list):
    reshaped_tensors = []

    for tensor in tensor_list:
        reshaped_tensor = tensor.view(-1)
        reshaped_tensors.append(reshaped_tensor)

    concatenated_tensors = torch.cat(reshaped_tensors)
    
    return concatenated_tensors
  
def hessian_vector_product(model, vector, input, target, loss_func, r=1e-2):
    R = r / flattener(vector).norm()
    for p, v in zip(model.parameters(), vector):
      p.data.add_(v, alpha=R)
    unrolled_output = model(input)
    unrolled_loss = loss_func(unrolled_output, target)
    grads_p = torch.autograd.grad(unrolled_loss, model.architecture_parameters)

    for p, v in zip(model.parameters(), vector):
      p.data.add_(v, alpha=R*2)
    unrolled_output = model(input)
    unrolled_loss = loss_func(unrolled_output, target)
    grads_n = torch.autograd.grad(unrolled_loss, model.architecture_parameters)

    for p, v in zip(model.parameters(), vector):
      p.data.add_(v, alpha=R)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
    
CLASSES = 10   

def main():
    logger.info("Starting architecture search")
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
  
    loss_func = torch.nn.CrossEntropyLoss().cuda()
    model = Supernet(args.channels, CLASSES, loss_func, num_cells=args.layers).cuda()

    network_optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    architecture_optimizer = torch.optim.Adam(
        model.architecture_parameters,
        lr=args.arch_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=args.arch_weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(network_optimizer, args.epochs, eta_min=args.learning_rate_min)
    
    for epoch in range(args.epochs):
      logger.info(f'epoch {epoch}')

      model.train()
      current_learning_rate = scheduler.get_last_lr()[0]
      top1 = AvgrageMeter()
      top5 = AvgrageMeter()

      for batch_no, (input, gt) in enumerate(train_loader):
        input_search, gt_search = next(iter(valid_loader))
        
        input = input.to('cuda')
        gt = gt.to('cuda')
        
        input_search = input_search.to('cuda')
        gt_search = gt_search.to('cuda')
        
        batch_size = input.size(0)

        architecture_optimizer.zero_grad()
        #only the unrolled model is implemented
        unrolled_output = model(input_search)
        unrolled_loss = loss_func(unrolled_output, gt_search)
        flattened_params = flattener(model.parameters()).data
        
        try:
          momentum_buffers = []

          for parameter in model.parameters():
              momentum_buffer = network_optimizer.state[parameter]['momentum_buffer']
              momentum_buffers.append(momentum_buffer)

          flattened_moments = flattener(momentum_buffers)
          flattened_moments.mul_(args.momentum)
          flattened_moments = flattened_moments.data
        except:
          flattened_moments = torch.zeros_like(flattened_params)
        
        decayed_parameters = args.arch_weight_decay * flattened_params
        #using autograd here instead of backwards to avoid saving network parameter gradients
        parameter_gradients = flattener(torch.autograd.grad(unrolled_loss, model.parameters())).data + decayed_parameters
        
        #construct the unrolled model
        unrolled_model = Supernet(args.channels, CLASSES, loss_func, num_cells=args.layers).cuda()
        for unrolled_arch_param, base_arch_param in zip(unrolled_model.architecture_parameters, model.architecture_parameters):
          unrolled_arch_param.data.copy_(base_arch_param.data)
        model_state = model.state_dict()
        new_params = flattened_params.sub((flattened_moments + parameter_gradients) * current_learning_rate)
        params, offset = {}, 0
        for param_name, param_value in model.named_parameters():
          num_params = np.prod(param_value.size())
          params[param_name] = new_params[offset: offset+num_params].view(param_value.size())
          offset += num_params
          
        assert offset == len(new_params)
        model_state.update(params)
        unrolled_model.load_state_dict(model_state)
        
        unrolled_output = unrolled_model(input)
        unrolled_loss = loss_func(unrolled_output, gt)
        
        unrolled_loss.backward()
        architecture_param_grads = []
        for v in unrolled_model.architecture_parameters:
            architecture_param_grads.append(v.grad)
            
        unrolled_params = []
        for unrolled_model_param in unrolled_model.parameters():
            unrolled_params.append(unrolled_model_param.grad.data)
            
        estimated_gradients = hessian_vector_product(model, unrolled_params, input_search, gt_search, loss_func)
        
        for arch_param_grad, estimated_grad in zip(architecture_param_grads, estimated_gradients):
          arch_param_grad.data.sub_(estimated_grad.data, alpha=current_learning_rate)


        for arch_parameter, arch_param_grad in zip(model.architecture_parameters, architecture_param_grads):
          if arch_parameter.grad is None:
            arch_parameter.grad = arch_param_grad.data.clone().detach().requires_grad_(True)
          else:
            arch_parameter.grad.data.copy_(arch_param_grad.data)
        
        architecture_optimizer.step()

        network_optimizer.zero_grad()
        output = model(input)
        output_loss = loss_func(output, gt)
        output_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # Use clip_grad_norm to avoid exploding gradients
        network_optimizer.step()


        top1new, top5new = accuracy(output, gt, topk=(1, 5))
        top1.update(top1new.item(), batch_size)
        top5.update(top5new.item(), batch_size)
        
        logger.info(f'train {batch_no} {top1.avg} {top5.avg}')

      scheduler.step()

      model.eval()
      top1 = AvgrageMeter()
      top5 = AvgrageMeter()

      for batch_no, (input, gt) in enumerate(valid_loader):
        with torch.no_grad():
          input = input.to('cuda')
          gt = gt.to('cuda')
          output = model(input)
          output_loss = loss_func(output, gt)

        prec1, prec5 = accuracy(output, gt, topk=(1, 5))
        n = input.size(0)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        logger.info(f'validation {batch_no} {top1.avg} {top5.avg}')


if __name__ == '__main__':
  main() 