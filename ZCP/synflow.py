import torch
import torch.nn as nn

def synflow_score(model, dataloader):

    def linearize(model):
        signs = {}
        for name, param in model.named_parameters():
            signs[name] = torch.sign(param.data).to('cuda')
            param.data = param.data.abs_()
        return signs

    def nonlinearize(model, signs):
        for name, param in model.named_parameters():
            param.data.mul_(signs[name])

    # linearize the model and save the signs
    signs = linearize(model)

    model.zero_grad()

    inputs, _ = next(iter(dataloader))
    inputs = inputs.to('cuda')

    # Forward and backward passes
    output = model.forward(inputs)
    torch.sum(output[0]).backward()

    def synflow(layer):
        if hasattr(layer, 'weight') and layer.weight is not None:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    return (layer.weight.data * layer.weight.grad.data).abs()
        return None

    layer_synflows = [s for s in (synflow(layer) for layer in model.modules()) if s is not None]
    
    total_synflow_score = torch.cat([synflow.view(-1) for synflow in layer_synflows])

    total_synflow_score = total_synflow_score.sum().item()

    # Restore the model's parameters
    nonlinearize(model, signs)

    model.zero_grad()

    return total_synflow_score