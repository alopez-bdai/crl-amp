from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *
import torch

from amp.dataloader.motion_loader import MotionLoader
from amp.envs.base.base_task import BaseTask
from amp.utils.helpers import class_to_dict
from amp.utils.math import get_quat_yaw, wrap_to_pi, quat_apply_yaw


class AMPTask(BaseTask):
    """ AMP Task: Simple AMP pipeline to follow joystick commands.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.amp_loader = MotionLoader(device=self.device,
                                       sim_frame_dt=self.dt,
                                       cfg=cfg.motion_loader,
                                       num_joints=cfg.env.num_actions,
                                       num_ee=len(cfg.asset.ee_offsets.keys())
                                       )

        # amp buffers
        self.amp_obs_buf = torch.zeros(self.num_envs, self.num_amp_obs, dtype=torch.float, device=self.device)
        self.amp_frame = torch.zeros(self.num_envs, self.amp_loader.amp_frame_dim, dtype=torch.float,
                                     device=self.device)
        self.last_amp_frames = torch.zeros(self.num_envs, self.cfg.motion_loader.num_amp_frames,
                                           self.amp_loader.amp_frame_dim, dtype=torch.float,
                                           device=self.device)
        self.reset_frames = torch.zeros(self.num_envs, self.amp_loader.full_frame_dim, device=self.device)

        # setup variables
        self.reset_triggered = True

        self.ref_state_init = self.cfg.env.reference_state_initialization
        self.add_noise = self.cfg.observations.add_noise
        self.change_commands = self.cfg.commands.change_commands
        self.push_robots = self.cfg.domain_rand.push_robots
        self.randomize_gains = self.cfg.domain_rand.randomize_gains

        if self.add_noise:
            self.noise_scale_vec = self._get_noise_scale_vec()

        if self.change_commands:
            self.change_commands_interval = np.ceil(self.cfg.commands.change_commands_interval_s / self.dt)

        if self.push_robots:
            self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

        if self.randomize_gains:
            self.randomized_p_gains, self.randomized_d_gains = self._compute_randomized_gains(self.num_envs)

        self._prepare_reward_function()
        self.init_done = True
        self._validate_config()

    def _validate_config(self):
        self.post_physics_step()
        assert self.obs_buf.size(dim=1) == self.num_obs

    def _init_buffers(self):
        super()._init_buffers()

        # setup variables
        self.play_step = 0

        # actuator
        self.actuator_lag_buffer = torch.zeros(self.num_envs, self.cfg.domain_rand.actuator_lag_steps + 1,
                                               self.num_actions, dtype=torch.float, device=self.device)
        self.actuator_lag_index = torch.randint(low=0, high=self.cfg.domain_rand.actuator_lag_steps,
                                                size=[self.num_envs], device=self.device)

        # end effector
        self.last_ee_local = torch.zeros(self.num_envs, self.num_ee, 3, dtype=torch.float, device=self.device)
        self.predicted_contact = torch.zeros(self.num_envs, self.num_ee, dtype=torch.bool, device=self.device)

        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.command_keys = [key for key in self.command_ranges.keys()]
        self.command_idx = {self.command_keys[i]: i for i in range(len(self.command_keys))}
        self.commands = torch.zeros(self.num_envs, len(self.command_ranges), dtype=torch.float,
                                    device=self.device)

        if self.cfg.motion_loader.preload_mode == 'trajectory':
            self.reset_state_traj_ids_buf = np.zeros(self.num_envs, dtype=np.int)  # idx of traj used for reset
            self.reset_state_traj_times_buf = np.zeros(self.num_envs, dtype=np.int)  # frame time in traj used for reset

    @property
    def num_amp_obs(self):
        return self.amp_loader.amp_obs_dim

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # update curriculum
        if self.cfg.env.play and self.cfg.terrain.mesh_type != 'plane':
            self._sample_terrain(env_ids)

        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_term_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_term_sums[key][env_ids]) / self.max_episode_length
            self.episode_term_sums[key][env_ids] = 0.

        for key in self.episode_group_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_group_sums[key][env_ids]) / self.max_episode_length
            self.episode_group_sums[key][env_ids] = 0.

        self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # reset robot states
        if self.ref_state_init:
            if self.cfg.motion_loader.preload_mode == 'trajectory':
                env_ids_np = env_ids.cpu().numpy()
                if self.cfg.env.play:
                    # pick random idx from total number of frames, then find the corresponding traj_idx and frame_idx
                    # random_idxs = np.random.randint(0, self.amp_loader.preload_traj_lens_cumsum[-1], size=len(env_ids))
                    # traj_idxs = self.amp_loader.preload_traj_lens_cumsum.searchsorted(random_idxs, 'right')
                    # frame_idxs = random_idxs - (
                    #         self.amp_loader.preload_traj_lens_cumsum[traj_idxs] - self.amp_loader.preload_traj_lens[
                    #     traj_idxs])
                    env_ids_np = env_ids.cpu().numpy()
                    idxs = np.arange(self.cfg.motion_loader.len_preload_buf)
                    traj_idxs = idxs[env_ids_np]
                    frame_idxs = np.random.randint(0, self.amp_loader.preload_traj_lens[traj_idxs])
                else:
                    idxs = np.arange(self.cfg.motion_loader.len_preload_buf)
                    traj_idxs = idxs[env_ids_np]
                    frame_idxs = np.random.randint(0, self.amp_loader.preload_traj_lens[traj_idxs])
                self.reset_state_traj_ids_buf[env_ids_np] = traj_idxs
                self.reset_state_traj_times_buf[env_ids_np] = frame_idxs
                frames = self.amp_loader.preload_full_buf[traj_idxs, frame_idxs]
            else:
                frames = self.amp_loader.get_full_frame_batch(len(env_ids))

            sample_choice = torch.rand((len(env_ids),), device=self.device) < self.cfg.env.rsi_ratio
            ids_frame = torch.where(sample_choice)[0]
            ids_default = torch.where(~sample_choice)[0]
            if len(env_ids[ids_frame]) > 0:
                self._reset_dofs_amp(env_ids[ids_frame], frames[ids_frame], env_ids[ids_default])
                self._reset_root_states_amp(env_ids[ids_frame], frames[ids_frame], env_ids_default=env_ids[ids_default])
                self.reset_frames[ids_frame] = frames[ids_frame]
                self.reset_frames[ids_default] = 0
            else:
                self._reset_dofs(env_ids[ids_default])
                self._reset_root_states(env_ids[ids_default])
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        self._refresh_quantities()

        if self.cfg.viewer.ref_env in env_ids:
            self.viewer_terrain_height = self._get_mean_height_around_pos(
                self.root_states[self.cfg.viewer.ref_env, :].unsqueeze(0), 2.0, num_points=11)
        if self.randomize_gains:
            new_randomized_gains = self._compute_randomized_gains(len(env_ids))
            self.randomized_p_gains[env_ids] = new_randomized_gains[0]
            self.randomized_d_gains[env_ids] = new_randomized_gains[1]

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_base_lin_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.last_ee_global[env_ids] = 0.
        self.last_ee_local[env_ids] = 0.
        self.last_amp_frames[env_ids] = 0.
        self.total_power[env_ids] = 0.
        self.total_torque[env_ids] = 0.
        self.actuator_lag_buffer[env_ids] = 0.

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()

    def _refresh_quantities(self):
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.ee_contact_force = torch.norm(self.contact_forces[:, self.ee_indices, :], dim=-1)
        self.ee_contact = self.ee_contact_force > 0.1
        ee_global_ = self.rigid_body_state[:, self.ee_indices, 0:3]
        ee_quat = self.rigid_body_state[:, self.ee_indices, 3:7]
        for i in range(len(self.ee_indices)):
            self.ee_global[:, i, :] = ee_global_[:, i, :] + quat_rotate(ee_quat[:, i, :],
                                                                        self.ee_offsets[i, :].unsqueeze(0).repeat(
                                                                            self.num_envs, 1))
            ee_local_ = self.ee_global[:, i, :].squeeze() - self.root_states[:, 0:3]
            self.ee_local[:, i, :] = quat_rotate_inverse(self.base_quat, ee_local_)

        self.joint_targets_rate = torch.norm(self.last_joint_targets - self.joint_targets, p=2,
                                             dim=1) / self.joint_targets_rate_scaler
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        self.base_lin_acc = (self.last_base_lin_vel - self.base_lin_vel) / self.dt

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()  # absolute terrain height

        self.amp_frame = self.get_current_amp_frame()

    def pre_physics_step(self, actions):
        self.actions = actions
        clip_joint_target = self.cfg.control.clip_joint_target
        scale_joint_target = self.cfg.control.scale_joint_target
        self.joint_targets = torch.clip(actions * scale_joint_target, -clip_joint_target, clip_joint_target).to(
            self.device)

    def step(self, actions):
        self.pre_physics_step(actions)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            if self.cfg.asset.default_dof_drive_mode == 3:  # torque mode
                self.torques = self._compute_torques(self.joint_targets).view(self.torques.shape)
                power_ = self.torques * self.dof_vel
                total_power_ = torch.sum(power_ * (power_ >= 0), dim=1)
                total_torque_ = torch.sum(torch.square(self.torques), dim=1)
                self.total_power += total_power_
                self.total_torque += total_torque_
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            if self.cfg.asset.default_dof_drive_mode == 1:  # position mode
                pos_targets = self.joint_targets + self.default_dof_pos
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_targets))

            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        return self.obs_buf, self.privileged_obs_buf, self.amp_obs_buf, self.group_rew_buf, self.reset_buf, self.extras

    def check_termination(self):
        # contact termination
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 0.1,
            dim=1)

        # reach maximal velocity termination
        self.reset_buf |= torch.norm(self.base_lin_vel, dim=-1) > self.cfg.termination.max_base_lin_vel
        self.reset_buf |= torch.norm(self.base_ang_vel, dim=-1) > self.cfg.termination.max_base_ang_vel
        rel_height = self.root_states[:, 2] - self._get_max_height_around_pos(self.root_states, 1.0)
        self.reset_buf |= torch.abs(rel_height) > self.cfg.termination.max_rel_height

        # reach joint limit termination
        if self.cfg.asset.terminate_on_joint_limit:
            pos_deviation = torch.abs(
                self.dof_pos - torch.clip(self.dof_pos, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1]))
            vel_deviation = torch.abs(
                self.dof_vel - torch.clip(self.dof_vel, -self.dof_vel_limits, self.dof_vel_limits))
            self.reset_buf |= torch.norm(pos_deviation, dim=-1) > 0.001
            self.reset_buf |= torch.norm(vel_deviation, dim=-1) > 0.001

        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs

        if not self.cfg.env.play:
            self.reset_buf |= self.time_out_buf

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.episode_length_buf += 1
        self.play_step += 1

        self._refresh_quantities()

        self.check_termination()
        self.compute_reward()

        if self.change_commands:
            env_ids_change_commands = (self.episode_length_buf % self.change_commands_interval == 0).nonzero(
                as_tuple=False).flatten()
            self._resample_commands(env_ids_change_commands)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        if self.push_robots and (self.play_step % self.push_interval == 0):
            self._push_robots()

        env_ids_reset = self.reset_buf.nonzero().flatten()
        self.reset_idx(env_ids_reset)

        self.compute_observations()
        self.last_amp_frames = self.last_amp_frames.roll(shifts=-1, dims=1)
        self.last_amp_frames[:, -1, :] = self.amp_frame[:]
        self.compute_amp_observations()

        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_joint_targets[:] = self.joint_targets[:]
        self.last_ee_global[:] = self.ee_global[:]
        self.last_ee_local[:] = self.ee_local[:]

    def get_observations(self):
        return self.obs_buf

    def compute_observations(self):
        base_height = self.root_states[:, 2].unsqueeze(1) - self._get_height_at_points(self.root_states.unsqueeze(1))
        self.obs_buf = torch.cat((self.base_lin_vel,
                                  self.base_ang_vel,
                                  (self.dof_pos - self.default_dof_pos),
                                  self.dof_vel,
                                  self.projected_gravity,
                                  self.actions,
                                  base_height,
                                  self.commands,
                                  ), dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = self.root_states[:, 2].unsqueeze(1) - self.measured_heights
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # clip the observation if needed
        if self.cfg.observations.clip_obs:
            self.obs_buf = torch.clip(self.obs_buf, -self.cfg.observations.clip_limit, self.cfg.observations.clip_limit)

    def get_current_amp_frame(self):
        rel_height = self.root_states[:, 2] - self._get_mean_height_around_pos(self.root_states, 0.1)
        amp_frame = torch.cat((self.dof_pos,
                               self.ee_local.flatten(start_dim=1),
                               self.base_lin_vel,
                               self.base_ang_vel,
                               self.dof_vel,
                               rel_height.unsqueeze(1),
                               ), dim=-1)
        return amp_frame

    def get_current_full_amp_frame(self):
        pos = self.root_states[:, :3]
        rot = self.root_states[:, 3:7]
        joint_pos = self.dof_pos
        ee_pos_local = self.ee_local.flatten(start_dim=1)
        lin_vel = self.base_lin_vel  # in base frame
        ang_vel = self.base_ang_vel  # in base frame
        joint_vel = self.dof_vel
        ee_vel_local = ((self.ee_local - self.last_ee_local) / self.dt).flatten(start_dim=1)
        frame = torch.cat([pos, rot, joint_pos, ee_pos_local, lin_vel, ang_vel, joint_vel, ee_vel_local], dim=-1)
        return frame

    def compute_amp_observations(self):
        self.amp_obs_buf = self.last_amp_frames.view(self.num_envs, -1)
        return

    def _resample_commands(self, env_ids):
        for i in range(len(self.command_keys)):
            self.commands[env_ids, i] = torch_rand_float(self.command_ranges[self.command_keys[i]][0],
                                                         self.command_ranges[self.command_keys[i]][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :] *= (torch.norm(self.commands[env_ids, :], dim=1) > 0.2).unsqueeze(1)

        if self.cfg.env.play:
            self.commands[env_ids, :] = 0.0

    def _update_dofs_for_reset(self, env_ids):
        if self.cfg.env.play:
            self.dof_pos[env_ids] = self.default_dof_pos
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dof),
                                                                            device=self.device)
        self.dof_vel[env_ids] = 0.

    def _reset_dofs(self, env_ids):
        self._update_dofs_for_reset(env_ids)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_dofs_amp(self, env_ids, frames, env_ids_default=None):
        self.dof_pos[env_ids] = self.amp_loader.get_joint_pose(frames)
        self.dof_vel[env_ids] = self.amp_loader.get_joint_vel(frames)

        if env_ids_default is not None and len(env_ids_default) > 0:
            self._update_dofs_for_reset(env_ids_default)
            env_ids = torch.cat([env_ids, env_ids_default])
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _update_root_states_for_reset(self, env_ids):
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        if self.cfg.terrain.mesh_type != 'plane' and self.cfg.terrain.randomize_robot_origins:
            self.root_states[env_ids, :2] += torch_rand_float(-3., 3., (len(env_ids), 2),
                                                              device=self.device)  # random planar position
        # add height
        self.root_states[env_ids, 2] += self._get_max_height_around_pos(self.root_states[env_ids, :], 0.1)

        # base velocities
        if self.cfg.env.play:
            self.root_states[env_ids, 7:13] = 0.0
        else:
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                               device=self.device)  # [7:10]: lin vel, [10:13]: ang vel

    def _reset_root_states(self, env_ids):
        self._update_root_states_for_reset(env_ids)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames, set_planar_pos=False, env_ids_default=None):
        root_pos = self.amp_loader.get_root_pos(frames)
        if set_planar_pos:
            root_pos[:, :2] += self.env_origins[env_ids, :2]
        else:
            root_pos[:, :2] = self.env_origins[env_ids, :2]
            if self.cfg.terrain.mesh_type != 'plane' and self.cfg.terrain.randomize_robot_origins:
                torch.rand(len(env_ids), 2, device=self.device) * 6.0 - 3.0
                root_pos[:, :2] += torch_rand_float(-3., 3., (len(env_ids), 2),
                                                    device=self.device)  # random planar position

        root_pos[:, 2] += self._get_max_height_around_pos(root_pos, 0.1)  # add height
        self.root_states[env_ids, :3] = root_pos
        root_orn = self.amp_loader.get_root_rot(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, self.amp_loader.get_linear_vel(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, self.amp_loader.get_angular_vel(frames))

        if env_ids_default is not None and len(env_ids_default) > 0:
            self._update_root_states_for_reset(env_ids_default)
            env_ids = torch.cat([env_ids, env_ids_default])
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        # base velocity impulse
        max_vel = self.cfg.domain_rand.max_push_vel_xyz
        self.root_states[:, 7:10] += torch_rand_float(-max_vel, max_vel, (self.num_envs, 3),
                                                      device=self.device)  # lin vel x/y/z
        max_avel = self.cfg.domain_rand.max_push_avel_xyz
        self.root_states[:, 10:13] += torch_rand_float(-max_avel, max_avel, (self.num_envs, 3),
                                                       device=self.device)  # ang vel x/y/z
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_scales = self.cfg.observations.noise_scales
        noise_level = self.cfg.observations.noise_level
        noise_vec[:3] = noise_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel
        noise_vec[6:18] = noise_scales.dof_pos
        noise_vec[18:30] = noise_scales.dof_vel
        noise_vec[30:33] = noise_scales.gravity
        return noise_vec * noise_level

    def _sample_terrain(self, env_ids):
        self.terrain_levels[env_ids] = torch.randint_like(self.terrain_levels[env_ids], high=self.max_terrain_level)
        self.terrain_types[env_ids] = torch.randint_like(self.terrain_types[env_ids],
                                                         high=self.cfg.terrain.num_cols)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return

        target_tracking_rew = self.episode_term_sums['lin_vel_x']

        move_up = (target_tracking_rew[env_ids] >= 0.6)
        move_down = (target_tracking_rew[env_ids] < 0.2) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minimum level is zero)

        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _draw_debug_vis(self, vis_flag='height_field', frame=None):
        """ Draws visualizations for debugging (slows down simulation a lot).
            vis_flag options: 'height_field', 'end_effector'.
            include 'ref_only' in vis_flag if you want to draw only for ref env
        """
        super()._draw_debug_vis(vis_flag=vis_flag)

        self.gym.clear_lines(self.viewer)
        focus_env = self.cfg.viewer.ref_env
        color_red = (1, 0, 0)
        color_green = (0, 1, 0)
        color_blue = (0, 0, 1)
        color_white = (0, 0, 0)
        color_black = (0.2, 0.2, 0.2)
        color_yellow = (1, 1, 0)

        if 'end_effector' in vis_flag:
            if frame is None:
                # draw global ee pos
                sphere_geom_green = gymutil.WireframeSphereGeometry(0.03, 4, 4, None, color=color_green)
                sphere_geom_red = gymutil.WireframeSphereGeometry(0.03, 4, 4, None, color=color_red)
                for env_id in range(self.num_envs):
                    if env_id == self.cfg.viewer.ref_env or 'ref_only' not in vis_flag:
                        ee_pos = self.ee_global[env_id, :]
                        for j in range(self.num_ee):
                            x = ee_pos[j, 0]
                            y = ee_pos[j, 1]
                            z = ee_pos[j, 2]
                            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                            if self.ee_contact[env_id, j]:
                                gymutil.draw_lines(sphere_geom_green, self.gym, self.viewer, self.envs[env_id],
                                                   sphere_pose)
                            else:
                                gymutil.draw_lines(sphere_geom_red, self.gym, self.viewer, self.envs[env_id],
                                                   sphere_pose)
            else:
                # draw ee pos from reference frame
                sphere_geom = gymutil.WireframeSphereGeometry(0.03, 4, 4, None, color=color_blue)
                root_pos = self.amp_loader.get_root_pos(frame)
                root_rot = self.amp_loader.get_root_rot(frame)
                ee_pos_local = self.amp_loader.get_ee_pos_local(frame).reshape((-1, self.num_ee, 3))
                for env_id in range(self.num_envs):
                    if env_id == self.cfg.viewer.ref_env or 'ref_only' not in vis_flag:
                        ee_pos = quat_rotate(root_rot[env_id, :].repeat(self.num_ee, 1), ee_pos_local[env_id, :]) \
                                 + root_pos[env_id, :].repeat(self.num_ee, 1)
                        for j in range(self.num_ee):
                            x = ee_pos[j, 0]
                            y = ee_pos[j, 1]
                            z = ee_pos[j, 2]
                            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)

        if 'ground_truth' in vis_flag:
            reference_frames = self.reset_frames[focus_env]
            reference_frames[2] += self.env_origins[focus_env, 2]
            pos = reference_frames[:2] - self.reset_frames[focus_env][:2]
            pos += self.env_origins[focus_env, :2]
            pos[0] -= 1.0  # shift x
            pos[1] += 0.0  # shift y
            for i in range(self.num_bodies):
                self.gym.set_rigid_body_color(self.envs[-1], 0, i, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(*color_green))
            self.env_origins[self.num_envs - 1] = self.env_origins[focus_env]
            self.draw_ghost_robot(self.num_envs - 1, torch.cat([pos, reference_frames[2:7]]),
                                  reference_frames[7:7 + self.num_dof])

    def render(self, sync_frame_time=True, replay=False):
        if not replay:
            self._draw_debug_vis(vis_flag=self.cfg.viewer.vis_flag)
        super().render(sync_frame_time=sync_frame_time)

    def process_keystroke(self):
        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.value > 0:
                if evt.action == "key_space":
                    self.is_playing = not self.is_playing
                if evt.action == "key_r":
                    self.reset_triggered = True
                if evt.action == "toggle_overview":
                    self.overview = not self.overview

                if evt.action == "key_plus":
                    self.cfg.viewer.ref_env = np.clip(self.cfg.viewer.ref_env + 1, 0, self.num_envs - 1)
                    print("focus env: ", self.cfg.viewer.ref_env)
                if evt.action == "key_minus":
                    self.cfg.viewer.ref_env = np.clip(self.cfg.viewer.ref_env - 1, 0, self.num_envs - 1)
                    print("focus env: ", self.cfg.viewer.ref_env)

                # todo: handle keyboard_increment index properly
                if evt.action == "key_up":
                    self.commands[:, self.command_idx['lin_vel_x']] += self.cfg.commands.keyboard_increment[0]
                if evt.action == "key_down":
                    self.commands[:, self.command_idx['lin_vel_x']] -= self.cfg.commands.keyboard_increment[0]
                if evt.action == "key_left":
                    self.commands[:, self.command_idx['ang_vel_z']] += \
                        self.cfg.commands.keyboard_increment[0]
                if evt.action == "key_right":
                    self.commands[:, self.command_idx['ang_vel_z']] -= \
                        self.cfg.commands.keyboard_increment[0]
                if evt.action == "key_a":
                    self.commands[:, self.command_idx['lin_vel_y']] += self.cfg.commands.keyboard_increment[0]
                if evt.action == "key_d":
                    self.commands[:, self.command_idx['lin_vel_y']] -= self.cfg.commands.keyboard_increment[0]
                # clip commands to be in range
                for i in range(len(self.command_keys)):
                    self.commands[:, i] = torch.clip(self.commands[:, i], self.command_ranges[self.command_keys[i]][0],
                                                     self.command_ranges[self.command_keys[i]][1])

    def get_time_stamp(self):
        return self.play_step * self.dt

    def reset_envs_to_frames(self, env_ids, frames):
        self._reset_dofs_amp(env_ids, frames)
        self._reset_root_states_amp(env_ids, frames, set_planar_pos=True)
        self._draw_debug_vis(vis_flag=self.cfg.viewer.vis_flag, frame=frames)

    def getplt_replay_vel(self):
        frame = self.amp_loader.get_full_frame_at_time(traj_idx=self.focus_env,
                                                       time=self.play_step * self.amp_loader.trajectory_frame_dt[
                                                           self.focus_env])
        fwd_vel = self.amp_loader.get_linear_vel(frame)[0:1].detach().cpu()
        label = "reference vel"
        legend = ["fwd_vel"]
        return np.array(fwd_vel), label, legend

    def getplt_vel_fwd(self):
        label = "forward vel"
        vel_fwd = self.base_lin_vel.detach().cpu()[self.focus_env, 0]
        if 'lin_vel_x' in self.command_keys:
            cmd_vel_fwd = self.commands.detach().cpu()[self.focus_env, self.command_idx['lin_vel_x']]
            out = [vel_fwd, cmd_vel_fwd]
            legend = ["actual", "cmd"]
        else:
            out = [vel_fwd]
            legend = ["actual"]
        return np.array(out), label, legend

    def getplt_vel_side(self):
        label = "side vel"
        vel_side = self.base_lin_vel.detach().cpu()[self.focus_env, 1]
        if 'lin_vel_y' in self.command_keys:
            cmd_vel_side = self.commands.detach().cpu()[self.focus_env, self.command_idx['lin_vel_y']]
            out = [vel_side, cmd_vel_side]
            legend = ["actual", "cmd"]
        else:
            out = [vel_side]
            legend = ["actual"]
        return np.array(out), label, legend

    def getplt_ang_vel(self):
        label = "angular vel"
        ang_vel = self.base_ang_vel.detach().cpu()[self.focus_env, 2]
        if 'ang_vel_z' in self.command_keys:
            cmd_ang_vel = self.commands.detach().cpu()[self.focus_env, self.command_idx['ang_vel_z']]
            out = [ang_vel, cmd_ang_vel]
            legend = ["actual", "cmd"]
        else:
            out = [ang_vel]
            legend = ["actual"]
        return np.array(out), label, legend

    def getplt_style_reward(self, disc_inference_func):
        def _get_styl_reward():
            focus_env = self.cfg.viewer.ref_env
            amp_obs = self.amp_obs_buf.detach()[focus_env]
            style_reward = disc_inference_func(amp_obs).detach().cpu().numpy()
            legend = ["style rew"]
            return style_reward, "style reward", legend

        return _get_styl_reward

    def _reward_lin_vel_x(self, sigma, tolerance=0.0):
        if 'lin_vel_x' in self.command_ranges:
            i = self.command_idx['lin_vel_x']
            lin_vel_x_error = self.commands[:, i] - self.base_lin_vel[:, 0]
        else:
            lin_vel_x_error = self.base_lin_vel[:, 0]
        lin_vel_x_error *= torch.abs(lin_vel_x_error) > tolerance  # ignore errors less than this threshold
        return torch.exp(-torch.square(lin_vel_x_error / sigma))

    def _reward_lin_vel_y(self, sigma, tolerance=0.0):
        if 'lin_vel_y' in self.command_ranges:
            i = self.command_idx['lin_vel_y']
            lin_vel_y_error = self.commands[:, i] - self.base_lin_vel[:, 1]
        else:
            lin_vel_y_error = self.base_lin_vel[:, 1]
        lin_vel_y_error *= torch.abs(lin_vel_y_error) > tolerance  # ignore errors less than this threshold
        return torch.exp(-torch.square(lin_vel_y_error / sigma))

    def _reward_ang_vel_z(self, sigma, tolerance=0.0):
        if 'ang_vel_z' in self.command_ranges:
            i = self.command_idx['ang_vel_z']
            ang_vel_error = self.commands[:, i] - self.base_ang_vel[:, 2]
        else:
            ang_vel_error = self.base_ang_vel[:, 2]
        ang_vel_error *= torch.abs(ang_vel_error) > tolerance  # ignore errors less than this threshold
        return torch.exp(-torch.square(ang_vel_error / sigma))

    def _reward_joint_targets_rate(self, sigma, tolerance=0.0):
        joint_rate_error = torch.abs(self.joint_targets_rate)
        joint_rate_error *= joint_rate_error > tolerance
        return torch.exp(-torch.square(joint_rate_error / sigma))

    def _reward_base_height(self, sigma, tolerance=0.0):
        # Penalize base height away from target
        height_error = self.root_states[:, 2] - self.cfg.rewards.base_height_target
        return torch.exp(-torch.square(height_error / sigma))

    def _reward_joint_soft_limits(self, sigma, tolerance=0.0):
        deviation = torch.maximum(self.dof_pos - self.dof_pos_soft_limits[:, 1],
                                  self.dof_pos_soft_limits[:, 0] - self.dof_pos)
        penalty = torch.norm(torch.maximum(deviation, torch.zeros_like(deviation)), dim=-1)
        return torch.exp(-torch.square(penalty / sigma))

    def _reward_stand_still(self, sigma, tolerance=0.0):
        norm_commands = torch.norm(self.commands, dim=1)
        stand_command = norm_commands < tolerance
        penalty = torch.norm(self.dof_vel) / self.num_dof * stand_command
        return torch.exp(-torch.square(penalty / sigma))

    def _reward_feet_height(self, sigma, tolerance=0.0):
        feet_height = self.ee_global[:, :, 2] - self._get_height_at_points(self.ee_global)
        err = self.cfg.rewards.feet_height_target - torch.minimum(feet_height, torch.ones_like(
            feet_height) * self.cfg.rewards.feet_height_target)
        err *= ~self.ee_contact
        return torch.exp(-torch.square(torch.norm(err, dim=-1) / sigma))
