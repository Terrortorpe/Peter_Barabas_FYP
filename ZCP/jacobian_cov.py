import torch
import torch.nn as nn
import numpy as np

def jacob_cov(net, dataloader, num_batches=1):
    # Ensuring the network is in evaluation mode
    net.eval()

    # Data acquisition
    dataloader_iter = iter(dataloader)
    data = [next(dataloader_iter) for _ in range(num_batches)]
    inputs = torch.cat([a for a, _ in data]).to('cuda')
    targets = torch.cat([b for _, b in data]).to('cuda')

    inputs.requires_grad_(True)
    y = net(inputs)[0]
    y.backward(torch.ones_like(y))
    jacob = inputs.grad.detach()

    # Reshaping the Jacobian and calculating score
    jacobs = jacob.reshape(jacob.size(0), -1).cpu().numpy()
    eps = 1e-7
    std_devs = np.std(jacobs, axis=1)
    jacobs[std_devs == 0] += eps
    corrs = np.corrcoef(jacobs)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    jc = -np.sum(np.log(v + k) + 1./(v + k))

    # Putting the network back into training mode
    net.train()

    return jc