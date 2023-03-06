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



def givens_rotation(r, x, transpose=False):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate
        transpose: whether to transpose the rotation matrix

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    if transpose:
        x_rot = givens[:, :, 0:1] * x - givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    else:
        x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


def quaternion_rotation(rotation, x, right = False, transpose = False):
    """rotate a batch of quaterions with by orthogonal rotation matrices

    Args:
        rotation (torch.Tensor): parameters that used to rotate other vectors [batch_size, rank]
        x (torch.Tensor): vectors that need to be rotated [batch_size, rank]
        right (bool): whether to rotate left or right
        transpose (bool): whether to transpose the rotation matrix

    Returns:
        rotated_vectors(torch.Tensor): rotated_results [batch_size, rank]
    """
    # ! it seems like calculate in this way will slow down the speed
    # turn to quaterion
    rotation = rotation.view(rotation.shape[0], -1, 4)
    rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True).clamp_min(1e-15)  # unify each quaterion of the rotation part
    p, q, r, s = rotation[:, :, 0], rotation[:, :, 1], rotation[:, :, 2], rotation[:, :, 3]
    if transpose:
        q, r, s = -q, -r ,-s # the transpose of this quaterion rotation matrix can be achieved by negating the q, r, s
    s_a, x_a, y_a, z_a = x.chunk(dim=1, chunks=4)
    
    if right:
        # right rotation
        # the original version used in QuatE is actually the right rotation
        rotated_s = s_a * p - x_a * q - y_a * r - z_a * s
        rotated_x = s_a * q + p * x_a + y_a * s - r * z_a
        rotated_y = s_a * r + p * y_a + z_a * q - s * x_a
        rotated_z = s_a * s + p * z_a + x_a * r - q * y_a
    else:
        # left rotation
        rotated_s = s_a * p - x_a * q - y_a * r - z_a * s
        rotated_x = s_a * q + x_a * p - y_a * s + z_a * r
        rotated_y = s_a * r + x_a * s + y_a * p - z_a * q 
        rotated_z = s_a * s - x_a * r + y_a * q + z_a * p 

    rotated_vectors = torch.cat([rotated_s, rotated_x, rotated_y, rotated_z], dim=-1)

    return rotated_vectors
    