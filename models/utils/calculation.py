'''some base calculations'''

import torch
from torch import Tensor 

def euc_distance(x: Tensor, y: Tensor, eval_mode=False) -> Tensor:
    """calculate eucidean distance

    Args:
        x (Tensor): shape:(N1, d), the x tensor 
        y (Tensor): shape (N2, d) if eval_mode else (N1, d), the y tensor
        eval_mode (bool, optional): whether or not use eval model. Defaults to False.

    Returns:
        if eval mode: (N1, N2)
        else: (N1, 1)
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)

    if eval_mode:
        y2 = y2.t()
        xy = x @ y.t()
    else:
        assert x.shape[0] == y.shape[0], 'The shape of x and y do not match.'
        xy = torch.sum(x * y, dim=-1, keepdim=True)

    return x2 + y2 - 2 * xy


