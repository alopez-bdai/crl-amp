"""Utility functions for processing motion clips."""

import numpy as np
from amp.dataloader import pose3d
from pybullet_utils import transformations


def split_into_chunks(input_trajs, max_traj_len):
    start_idx = []
    lens = []
    for trajs in input_trajs:
        if len(trajs) > max_traj_len:
            # only add one chunk per sampled long trajectories #todo: make better
            start_idx.append(np.random.randint(low=0, high=len(trajs) - max_traj_len))
            lens.append(max_traj_len)
        else:
            start_idx.append(0)
            lens.append(len(trajs))
    return start_idx, lens


def standardize_quaternion(q):
    """Returns a quaternion where q.w >= 0 to remove redundancy due to q = -q.

    Args:
      q: A quaternion to be standardized.

    Returns:
      A quaternion with q.w >= 0.

    """
    if q[-1] < 0:
        q = -q
    return q


def normalize_rotation_angle(theta):
    """Returns a rotation angle normalized between [-pi, pi].

    Args:
      theta: angle of rotation (radians).

    Returns:
      An angle of rotation normalized between [-pi, pi].

    """
    norm_theta = theta
    if np.abs(norm_theta) > np.pi:
        norm_theta = np.fmod(norm_theta, 2 * np.pi)
        if norm_theta >= 0:
            norm_theta += -2 * np.pi
        else:
            norm_theta += 2 * np.pi

    return norm_theta


def calc_heading(q):
    """Returns the heading of a rotation q, specified as a quaternion.

    The heading represents the rotational component of q along the vertical
    axis (z axis).

    Args:
      q: A quaternion that the heading is to be computed from.

    Returns:
      An angle representing the rotation about the z axis.

    """
    ref_dir = np.array([1, 0, 0])
    rot_dir = pose3d.QuaternionRotatePoint(ref_dir, q)
    heading = np.arctan2(rot_dir[1], rot_dir[0])
    return heading


def calc_heading_rot(q):
    """Return a quaternion representing the heading rotation of q along the vertical axis (z axis).

    Args:
      q: A quaternion that the heading is to be computed from.

    Returns:
      A quaternion representing the rotation about the z axis.

    """
    heading = calc_heading(q)
    q_heading = transformations.quaternion_about_axis(heading, [0, 0, 1])
    return q_heading
