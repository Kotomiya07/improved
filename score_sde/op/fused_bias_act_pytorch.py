import torch
import torch.nn.functional as F

def fused_bias_act(input, bias=None, refer=None, act=1, grad=0, alpha=0.2, scale=1.0):
    """
    Fused bias activation function implemented in PyTorch.
    
    Args:
        input (torch.Tensor): Input tensor.
        bias (torch.Tensor): Bias tensor. Default is None.
        refer (torch.Tensor): Reference tensor. Default is None.
        act (int): Activation type. Default is 1.
        grad (int): Gradient computation flag. Default is 0.
        alpha (float): Alpha parameter for leaky ReLU. Default is 0.2.
        scale (float): Scaling factor. Default is 1.0.
    
    Returns:
        torch.Tensor: Result after applying fused bias and activation.
    """
    x = input
    if bias is not None:
        # Apply bias
        x = x + bias.view(1, -1, *([1] * (x.ndim - 2)))
    
    # Apply activation function
    if act == 1:  # Linear
        y = x
    elif act == 3:  # Leaky ReLU
        if grad == 0:
            y = F.leaky_relu(x, negative_slope=alpha)
        elif grad == 1:
            y = F.leaky_relu(refer, negative_slope=alpha) * x
        else:
            y = torch.zeros_like(x)
    else:
        raise ValueError(f"Unsupported activation type: {act}")
    
    # Scale the output
    y = y * scale
    
    return y
