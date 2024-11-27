import json
import warnings
import torch
import numpy as np
from isaacgym.torch_utils import quat_rotate

from amp.utils.helpers import get_paths_from_pattern
from amp.utils.math import quaternion_slerp, slerp
from amp import ROOT_DIR
from amp.dataloader import pose3d
from amp.dataloader import motion_util


class MotionLoader:

    def __init__(
            self,
            device,
            cfg,
            sim_frame_dt,
            num_joints=12,
            num_ee=4,
    ):
        """ Loads expert motion dataset and provides AMP observations. """

        self.num_joints = num_joints
        self.num_ee = num_ee

        self.cfg = cfg
        self.device = device
        self.sim_frame_dt = sim_frame_dt
        self.preload_mode = cfg.preload_mode
        self.num_amp_frames = cfg.num_amp_frames
        self.len_preload_buf = cfg.len_preload_buf

        self._init_indices()
        self._init_trajectory_data()
        motion_paths = get_paths_from_pattern(cfg.motion_files)
        assert len(motion_paths) > 0, f"No motion files found here: {cfg.motion_files}"
        for i, motion_path in enumerate(motion_paths):
            self._load_motion_file(motion_path, i)

        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_dt = np.array(self.trajectory_frame_dt)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload
        self.preload_buf = None
        self.preload_full_buf = None
        # NOTE: preload_full has different shape in transition and trajectory mode.
        # It is optimized to not store redundant data in trajectory mode.
        if self.preload_mode == 'transition':
            self.preload_transitions()
        elif self.preload_mode == 'trajectory':
            self.preload_trajectories()
        else:
            raise Exception("Preload mode not defined.")

        print(f'Finished preloading')

    def _init_trajectory_data(self):
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_dt = []
        self.trajectory_num_frames = []

    def _load_motion_file(self, motion_file, i):
        self.trajectory_names.append(motion_file.split('.')[0])
        self.trajectory_idxs.append(i)

        with open(motion_file, "r") as f:
            motion_json = json.load(f)
        motion_data = np.array(motion_json["Frames"])
        if motion_json.get("OutputMode", "Pybullet") != "ISAAC":
            raise Exception('loading motions with different orders is deprecated.')
        motion_data = self.normalize_root_quaternions(motion_data)

        # full frame trajectory
        full_frames = self.get_full_frames_from_motion_data(motion_data, motion_json)
        self.trajectories_full.append(full_frames)

        # trajectory weight: used to sample some trajectories more than others.
        weight = float(motion_json["MotionWeight"])
        weight = self.cfg.special_motion_weight if weight > 1.0 else weight
        self.trajectory_weights.append(weight)

        frame_dt = float(motion_json["FrameDuration"])
        self.trajectory_frame_dt.append(frame_dt)

        traj_len = (motion_data.shape[0] - 1) * frame_dt
        self.trajectory_lens.append(traj_len)
        self.trajectory_num_frames.append(int(motion_data.shape[0]))

        print(f"{i}: Loaded {traj_len}s. motion from {motion_file}.")

    def normalize_root_quaternions(self, motion_data):
        for frame_i in range(motion_data.shape[0]):
            root_rot = self.get_root_rot(motion_data[frame_i])
            root_rot = pose3d.QuaternionNormalize(root_rot)
            root_rot = motion_util.standardize_quaternion(root_rot)
            motion_data[frame_i, self.POS_SIZE:(self.POS_SIZE + self.ROT_SIZE)] = root_rot
        return motion_data

    def get_full_frames_from_motion_data(self, motion_data, motion_json):
        """
        full frame:
              pos(3), rot(4), joint_pose(num_joints), ee_pos_local(3 x num_ee),...
              ... linear_vel(3), angular_vel(3), joint_vel(num_joints)
        """
        full_frames = torch.tensor(motion_data[:, :self.EE_VEL_LOCAL_END_IDX], dtype=torch.float32, device=self.device)

        assert full_frames.shape[-1] == self.full_frame_dim
        return full_frames

    def preload_transitions(self):
        print(f'Preloading {self.len_preload_buf} transitions')
        if self.len_preload_buf < 10000:
            warnings.warn("Preloading less than 10000 transitions is not recommended.")

        traj_idxs = self.weighted_traj_idx_sample_batch(self.len_preload_buf)
        times = self.traj_time_sample_batch(traj_idxs)

        traj_idxs = traj_idxs.repeat(self.num_amp_frames)
        times = times.repeat(self.num_amp_frames)
        times_addition = np.tile(np.arange(0, self.num_amp_frames), self.len_preload_buf) * self.sim_frame_dt

        full_frame_buf = self.get_full_frame_at_time_batch(traj_idxs, times + times_addition).reshape(-1,
                                                                                                      self.num_amp_frames,
                                                                                                      self.full_frame_dim)
        self.preload_full_buf = full_frame_buf.reshape(-1, self.num_amp_frames * self.full_frame_dim)
        self.preload_buf = self.get_amp_frames(full_frame_buf).reshape(-1, self.num_amp_frames * self.amp_frame_dim)

    def preload_trajectories(self):
        print(f'Preloading {self.len_preload_buf} trajectories')
        if self.len_preload_buf > 10000:
            warnings.warn("Preloading more than 10000 trajectories is not recommended.")
        for dt in self.trajectory_frame_dt:
            if abs(dt - self.sim_frame_dt) > 0.001:
                warnings.warn(
                    "Frame dt is not same as simulation dt! Preloading Trajectories might be wrong."
                    "Try preloading transitions instead.")

        max_traj_len = self.cfg.max_len_preload_trajs
        self.preload_traj_idx = self.weighted_traj_idx_sample_batch(self.len_preload_buf).tolist()

        full_traj_list = [self.trajectories_full[idx] for idx in self.preload_traj_idx]
        start_idx, lens = motion_util.split_into_chunks(full_traj_list, max_traj_len + self.num_amp_frames - 1)
        chunked_full_trajs = [full_traj_list[i][start_idx[i]:start_idx[i] + lens[i]] for i in range(len(start_idx))]
        chunked_transition_trajs = self.get_amp_transition(chunked_full_trajs)

        self.preload_traj_lens = np.array([len(traj) for traj in chunked_transition_trajs], dtype=np.int)
        self.preload_traj_lens_cumsum = np.cumsum(self.preload_traj_lens)

        self.preload_buf = torch.zeros(self.len_preload_buf, max_traj_len, self.num_amp_frames * self.amp_frame_dim,
                                       device=self.device, dtype=torch.float32)
        self.preload_full_buf = torch.zeros(self.len_preload_buf, max_traj_len, self.full_frame_dim,
                                            device=self.device, dtype=torch.float32)
        for idx in range(len(chunked_full_trajs)):
            self.preload_full_buf[idx, :self.preload_traj_lens[idx], :] = \
                chunked_full_trajs[idx][
                :self.preload_traj_lens[idx]]  # last n frames not used to have same size as preload_buf
            self.preload_buf[idx, 0:self.preload_traj_lens[idx], :] = chunked_transition_trajs[idx]

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = np.maximum(self.sim_frame_dt, self.trajectory_frame_dt[traj_idx]) * self.cfg.num_amp_frames
        return max(0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = np.maximum(self.sim_frame_dt, self.trajectory_frame_dt[traj_idxs]) * self.cfg.num_amp_frames
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def get_amp_frames(self, full_frames):
        """ Extracts AMP frame from full frame.
            amp_frame:
                joint_pose(num_joints), ee_pos_local(3 * num_ee), linear_vel(3), angular_vel(3),...
                joint_vel(num_joints), base_height(1)
        """
        amp_frames = torch.cat([
            full_frames[..., self.JOINT_POSE_START_IDX:self.JOINT_VEL_END_IDX],
            full_frames[..., self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]
        ], dim=-1)

        return amp_frames

    def get_amp_transition(self, trajs):
        """
            Args:
                trajs: list of input trajectories containing full frame information
            Returns: list of transition trajectories containing transitions of amp frame
        """
        transition_buf = []
        for traj_i in range(len(trajs)):
            traj_frames = trajs[traj_i].unfold(step=1, size=self.num_amp_frames,
                                               dimension=0).transpose(1, 2)
            amp_frames = self.get_amp_frames(traj_frames)
            transition_buf.append(amp_frames.reshape(-1, self.num_amp_frames * self.amp_frame_dim))
        return transition_buf

    def get_ee_pos_global(self, frame):
        root_pos = self.get_root_pos(frame)
        root_rot = self.get_root_rot(frame)
        ee_pos_local = self.get_ee_pos_local(frame).reshape((-1, self.num_ee, 3))
        ee_pos = quat_rotate(root_rot.unsqueeze(1).repeat(1, self.num_ee, 1).reshape(-1, self.num_ee),
                               ee_pos_local.reshape(-1, 3)).reshape(-1, self.num_ee, 3) + root_pos.unsqueeze(1).repeat(1, self.num_ee, 1)
        return ee_pos

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frames(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        """ Returns full frame for the given trajectories at the specified times."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        blended_frames = torch.zeros(len(traj_idxs), self.full_frame_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            blended_frames[traj_mask] = self.blend_frames(trajectory[idx_low[traj_mask]],
                                                          trajectory[idx_high[traj_mask]],
                                                          blend[traj_mask])
        return blended_frames

    def get_full_frame_batch(self, num_frames):
        """ Returns random full frame batch. """
        traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
        times = self.traj_time_sample_batch(traj_idxs)
        return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frames(self, frame0, frame1, blend):
        """ Linearly interpolate between two full frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """
        root_pos0, root_pos1 = self.get_root_pos(frame0), self.get_root_pos(frame1)
        root_rot0, root_rot1 = self.get_root_rot(frame0), self.get_root_rot(frame1)
        extras0 = frame0[..., self.JOINT_POSE_START_IDX:self.full_frame_dim]
        extras1 = frame1[..., self.JOINT_POSE_START_IDX:self.full_frame_dim]

        blend_root_pos = slerp(root_pos0, root_pos1, blend)
        blend_root_rot = quaternion_slerp(root_rot0, root_rot1, blend)
        blend_extras = slerp(extras0, extras1, blend)
        return torch.cat([blend_root_pos, blend_root_rot, blend_extras], dim=-1)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_mode == 'transition':
                idxs = np.random.choice(self.preload_buf.shape[0], size=mini_batch_size)
                s = self.preload_buf[idxs, :]
            elif self.preload_mode == 'trajectory':
                traj_idxs = np.random.choice(self.preload_buf.shape[0], size=mini_batch_size)
                frame_idxs = np.random.randint(low=0, high=self.preload_traj_lens[traj_idxs])
                s = self.preload_buf[traj_idxs, frame_idxs]
            else:
                raise NotImplementedError
            yield s

    @property
    def amp_frame_dim(self):
        """Size of a single AMP frame."""
        return self.JOINT_VEL_END_IDX - self.JOINT_POSE_START_IDX + 1

    @property
    def amp_obs_dim(self):
        """Size of AMP observation"""
        return self.cfg.num_amp_frames * self.amp_frame_dim

    @property
    def full_frame_dim(self):
        """Size of full frame observations."""
        return self.EE_VEL_LOCAL_END_IDX

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(self, pose):
        return pose[..., self.ROOT_POS_START_IDX:self.ROOT_POS_END_IDX]

    def get_root_rot(self, pose):
        return pose[..., self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]

    def get_joint_pose(self, pose):
        return pose[..., self.JOINT_POSE_START_IDX:self.JOINT_POSE_END_IDX]

    def get_ee_pos_local(self, pose):
        return pose[..., self.EE_POS_LOCAL_START_IDX:self.EE_POS_LOCAL_END_IDX]

    def get_linear_vel(self, pose):
        return pose[..., self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX]

    def get_angular_vel(self, pose):
        return pose[..., self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX]

    def get_joint_vel(self, pose):
        return pose[..., self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX]

    def get_ee_vel_local(self, pose):
        return pose[..., self.EE_VEL_LOCAL_START_IDX:self.EE_VEL_LOCAL_END_IDX]


    def _init_indices(self):
        # IsaacGym order[FL, FR, RL, RR].
        self.POS_SIZE = 3  # World frame
        self.ROT_SIZE = 4  # World frame
        self.JOINT_POS_SIZE = self.num_joints
        self.EE_POS_LOCAL_SIZE = 3 * self.num_ee  # Local frame
        self.LINEAR_VEL_SIZE = 3  # Local frame
        self.ANGULAR_VEL_SIZE = 3  # Local frame
        self.JOINT_VEL_SIZE = self.num_joints
        self.EE_VEL_LOCAL_SIZE = 3 * self.num_ee  # Local frame, EE vel is not very trustworthy

        self.ROOT_POS_START_IDX = 0
        self.ROOT_POS_END_IDX = self.ROOT_POS_START_IDX + self.POS_SIZE

        self.ROOT_ROT_START_IDX = self.ROOT_POS_END_IDX
        self.ROOT_ROT_END_IDX = self.ROOT_ROT_START_IDX + self.ROT_SIZE

        self.JOINT_POSE_START_IDX = self.ROOT_ROT_END_IDX
        self.JOINT_POSE_END_IDX = self.JOINT_POSE_START_IDX + self.JOINT_POS_SIZE

        self.EE_POS_LOCAL_START_IDX = self.JOINT_POSE_END_IDX
        self.EE_POS_LOCAL_END_IDX = self.EE_POS_LOCAL_START_IDX + self.EE_POS_LOCAL_SIZE

        self.LINEAR_VEL_START_IDX = self.EE_POS_LOCAL_END_IDX
        self.LINEAR_VEL_END_IDX = self.LINEAR_VEL_START_IDX + self.LINEAR_VEL_SIZE

        self.ANGULAR_VEL_START_IDX = self.LINEAR_VEL_END_IDX
        self.ANGULAR_VEL_END_IDX = self.ANGULAR_VEL_START_IDX + self.ANGULAR_VEL_SIZE

        self.JOINT_VEL_START_IDX = self.ANGULAR_VEL_END_IDX
        self.JOINT_VEL_END_IDX = self.JOINT_VEL_START_IDX + self.JOINT_VEL_SIZE

        self.EE_VEL_LOCAL_START_IDX = self.JOINT_VEL_END_IDX
        self.EE_VEL_LOCAL_END_IDX = self.EE_VEL_LOCAL_START_IDX + self.EE_VEL_LOCAL_SIZE
