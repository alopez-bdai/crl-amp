from typing import Tuple

import numpy as np
import torch
from isaacgym.torch_utils import quat_apply, normalize
from torch import Tensor


# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


def get_quat_yaw(quat):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_yaw


# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.) / 2.
    return (upper - lower) * r + lower


@torch.jit.script
def slerp(val0, val1, blend):
    return (1.0 - blend) * val0 + blend * val1


# @torch.jit.script
def quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """Batch quaternion spherical linear interpolation."""

    _EPS = torch.finfo(torch.float32).eps * 4.0

    out = torch.zeros_like(q0)

    zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
    ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
    out[zero_mask] = q0[zero_mask]
    out[ones_mask] = q1[ones_mask]

    d = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
    out[dist_mask] = q0[dist_mask]

    if shortestpath:
        d_old = torch.clone(d)
        d = torch.where(d_old < 0, -d, d)
        q1 = torch.where(d_old < 0, -q1, q1)

    precision_error = 0.00001
    if torch.any(d.abs() > 1.0 + precision_error):
        raise ValueError(f"Error in Quaternion SLERP. Argument to acos is larger than {1.0 + precision_error}.")
    else:
        d = torch.clip(d, -1.0, 1.0)

    angle = torch.acos(d) + spin * torch.pi
    angle_mask = (torch.abs(angle) < _EPS).squeeze()
    out[angle_mask] = q0[angle_mask]

    final_mask = torch.logical_or(zero_mask, ones_mask)
    final_mask = torch.logical_or(final_mask, dist_mask)
    final_mask = torch.logical_or(final_mask, angle_mask)
    final_mask = torch.logical_not(final_mask)

    isin = 1.0 / angle
    final = q0 * (torch.sin((1.0 - fraction) * angle) * isin) + q1 * (torch.sin(fraction * angle) * isin)
    out[final_mask] = final[final_mask]
    return out


def bezier(t, points):
    # Bézier curve calculation
    n = len(points) - 1
    result = torch.zeros_like(points[0])
    for i in range(n + 1):
        result += (points[i] * np.math.comb(n, i) * ((1 - t) ** (n - i)) * (t ** i))
    return result


# @ torch.jit.script
def cubic_bezier(t, ctrl_pts):
    # Bézier curve calculation
    # t: [batch_size] query points
    # ctrl_pts: [batch_size, 4, num_dims]
    t = t.unsqueeze(-1)
    sqz = False
    if len(ctrl_pts.shape) == 2:
        sqz = True
        ctrl_pts = ctrl_pts.unsqueeze(0)

    pt = torch.pow(1 - t, 3) * ctrl_pts[:, 0] + \
         3 * torch.pow(1 - t, 2) * t * ctrl_pts[:, 1] + \
         3 * (1 - t) * torch.pow(t, 2) * ctrl_pts[:, 2] + \
         torch.pow(t, 3) * ctrl_pts[:, 3]

    if sqz:
        pt = pt.squeeze()
    return pt


# @ torch.jit_script
def cubic_bezier_deriv(t, ctrl_pts):
    # Bézier curve calculation
    # t: [batch_size] query points
    # ctrl_pts: [batch_size, 4, num_dims]
    t = t.unsqueeze(-1)
    sqz = False
    if len(ctrl_pts.shape) == 2:
        sqz = True
        ctrl_pts = ctrl_pts.unsqueeze(0)

    pt = -3 * torch.pow(1 - t, 2) * ctrl_pts[:, 0] + \
         3 * torch.pow(1 - t, 2) * ctrl_pts[:, 1] - \
         6 * (1 - t) * t * ctrl_pts[:, 1] + \
         6 * (1 - t) * t * ctrl_pts[:, 2] - \
         3 * torch.pow(t, 2) * ctrl_pts[:, 2] + \
         3 * torch.pow(t, 2) * ctrl_pts[:, 3]

    if sqz:
        pt = pt.squeeze()
    return pt


# @torch.jit.script
def torch_rand_float_ring(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return torch.sqrt((upper ** 2 - lower ** 2) * torch.rand(*shape, device=device) + lower ** 2)
