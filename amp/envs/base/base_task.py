from isaacgym.torch_utils import to_torch, torch_rand_float, get_axis_params, quat_rotate_inverse
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import sys
import torch
import os
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import deque

from amp import ROOT_DIR
from amp.envs.base.observation_buffer import ObservationBuffer
from amp.utils.helpers import class_to_dict
from amp.utils.terrain import Terrain
from amp.utils.math import quat_apply_yaw, bezier


# Base class for RL tasks
class BaseTask:

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        self.headless = headless
        self.is_playing = False
        self.init_done = False

        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        self.graphics_device_id = self.sim_device_id
        if self.headless:
            self.graphics_device_id = -1

        # parse env config
        self._parse_cfg()

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id,
                                       self.graphics_device_id,
                                       self.physics_engine,
                                       self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
            self.cfg.terrain.curriculum = False
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        else:
            raise Exception

        self._create_envs()
        self.gym.prepare_sim(self.sim)

        self.enable_viewer_sync = True
        self.overview = self.cfg.viewer.overview
        self.focus_env = self.cfg.viewer.ref_env
        self.viewer = None
        self.debug_viz = False

        self._init_buffers()
        self.viewer_terrain_height = self.env_origins[self.focus_env, 2]
        self._set_camera_recording()

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.cfg.viewer.enable_viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.overview:
                self._set_camera(self.cfg.viewer.overview_pos, self.cfg.viewer.overview_lookat)
            else:
                ref_pos = [self.root_states[self.focus_env, 0].item() + self.cfg.viewer.ref_pos_b[0],
                           self.root_states[self.focus_env, 1].item() + self.cfg.viewer.ref_pos_b[1],
                           self.cfg.viewer.ref_pos_b[2]]
                ref_lookat = [self.root_states[self.focus_env, 0].item(),
                              self.root_states[self.focus_env, 1].item(),
                              0.2]
                self._set_camera(ref_pos, ref_lookat)
            self.viewer_set = True

            # keyboard events
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "toggle_overview")
            # if running with a viewer, and we are running play function
            if not self.headless and self.cfg.env.play:
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "key_up")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "key_down")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "key_left")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "key_right")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "key_a")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "key_d")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "key_space")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "key_r")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "key_s")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_ADD, "key_plus")
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_NUMPAD_SUBTRACT, "key_minus")

    def _set_camera_recording(self):
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = self.cfg.viewer.camera_horizontal_fov
        camera_props.width = self.cfg.viewer.camera_width
        camera_props.height = self.cfg.viewer.camera_height
        self.camera_props = camera_props
        self.image_env = self.cfg.viewer.camera_env
        self.camera_sensors = self.gym.create_camera_sensor(self.envs[self.image_env], self.camera_props)
        self.camera_image = np.zeros((self.camera_props.height, self.camera_props.width, 4), dtype=np.uint8)

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError

    def compute_observations(self):
        raise NotImplementedError

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def render(self, sync_frame_time=True):
        # fetch results
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

        if self.cfg.viewer.record_camera_imgs:
            ref_pos = [self.root_states[self.image_env, 0].item() + self.cfg.viewer.camera_pos_b[0],
                       self.root_states[self.image_env, 1].item() + self.cfg.viewer.camera_pos_b[1],
                       self.cfg.viewer.camera_pos_b[2] + + self.env_origins[self.focus_env, 2]]
            ref_lookat = [self.root_states[self.image_env, 0].item(),
                          self.root_states[self.image_env, 1].item(),
                          0.2 + + self.env_origins[self.focus_env, 2]]
            cam_pos = gymapi.Vec3(ref_pos[0], ref_pos[1], ref_pos[2])
            cam_target = gymapi.Vec3(ref_lookat[0], ref_lookat[1], ref_lookat[2])

            self.gym.set_camera_location(self.camera_sensors, self.envs[self.image_env], cam_pos, cam_target)

        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "toggle_overview" and evt.value > 0:
                    self.overview = not self.overview
                    self.viewer_set = False

            # step graphics
            if self.enable_viewer_sync:
                if self.cfg.env.play and not self.overview:
                    ref_pos = [self.root_states[self.focus_env, 0].item() + self.cfg.viewer.ref_pos_b[0],
                               self.root_states[self.focus_env, 1].item() + self.cfg.viewer.ref_pos_b[1],
                               self.cfg.viewer.ref_pos_b[2] + self.viewer_terrain_height]
                    ref_lookat = [self.root_states[self.focus_env, 0].item(),
                                  self.root_states[self.focus_env, 1].item(),
                                  0.2 + self.viewer_terrain_height]
                    self._set_camera(ref_pos, ref_lookat)
                else:
                    if not self.viewer_set:
                        if self.overview:
                            self._set_camera(self.cfg.viewer.overview_pos, self.cfg.viewer.overview_lookat)
                        else:
                            ref_pos = [
                                self.root_states[self.focus_env, 0].item() + self.cfg.viewer.ref_pos_b[0],
                                self.root_states[self.focus_env, 1].item() + self.cfg.viewer.ref_pos_b[1],
                                self.cfg.viewer.ref_pos_b[2] + self.viewer_terrain_height]
                            ref_lookat = [self.root_states[self.focus_env, 0].item(),
                                          self.root_states[self.focus_env, 1].item(),
                                          0.2 + self.viewer_terrain_height]
                            self._set_camera(ref_pos, ref_lookat)
                        self.viewer_set = True
            else:
                self.gym.poll_viewer_events(self.viewer)

        if self.cfg.viewer.record_camera_imgs or (self.viewer and self.enable_viewer_sync):
            self.gym.step_graphics(self.sim)

            if self.cfg.viewer.record_camera_imgs:
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                self.camera_image = self.gym.get_camera_image(self.sim, self.envs[self.image_env], self.camera_sensors,
                                                              gymapi.IMAGE_COLOR).reshape((self.camera_props.height,
                                                                                           self.camera_props.width, 4))
                self.gym.end_access_image_tensors(self.sim)

            if self.viewer and self.enable_viewer_sync:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)

    # ------------------------------------------------------------------------------------------------------------------

    def _parse_cfg(self):
        self.num_envs = self.cfg.env.num_envs
        self.num_obs = self.cfg.env.num_observations
        self.num_privileged_obs = self.cfg.env.num_privileged_obs
        self.include_history_steps = self.cfg.env.include_history_steps
        self.num_actions = self.cfg.env.num_actions
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.max_episode_length = self.cfg.env.episode_length

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.transpose().flatten(), hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.terrain.cfg.static_friction
        tm_params.dynamic_friction = self.terrain.cfg.dynamic_friction
        tm_params.restitution = self.terrain.cfg.restitution

        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(ROOT_DIR=ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self._set_default_dof_pos()
        ee_names = [s for s in body_names if any(key in s for key in self.cfg.asset.ee_offsets.keys())]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + \
                               self.cfg.init_state.rot + \
                               self.cfg.init_state.lin_vel + \
                               self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim,
                                             gymapi.Vec3(0., 0., 0.),
                                             gymapi.Vec3(0., 0., 0.),
                                             int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            # pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.ee_indices = torch.zeros(len(ee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.ee_offsets = torch.zeros(len(ee_names), 3, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(len(ee_names)):
            self.ee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], ee_names[i])
            self.ee_offsets[i, :] = to_torch(self.cfg.asset.ee_offsets[ee_names[i]])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _process_rigid_shape_props(self, props, env_id):
        # randomize friction
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_rigid_body_props(self, props):
        # randomize mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.dof_pos_soft_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device)
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_soft_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_soft_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                # Note: position mode needs stiffness and damping to be set,
                # in velocity mode stiffness should be zero,
                # in torque mode both should be zero
                if props["driveMode"][i] == 1:  # position mode
                    props["stiffness"][i] = self.p_gains[i]
                    props["damping"][i] = self.d_gains[i]
                elif props["driveMode"][i] == 2:  # velocity mode
                    props["damping"][i] = self.d_gains[i]
                elif props["driveMode"][i] == 3:  # torque mode
                    props["damping"][i] = 0.01  # slightly positive for better stability

        return props

    def _set_default_dof_pos(self):
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _init_buffers(self):
        # basic buffers
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history = ObservationBuffer(self.num_envs, self.num_obs,
                                                     self.include_history_steps, self.device)

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.base_quat = self.root_states[:, 3:7]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # contact_forces shape: num_envs, num_bodies, xyz axis
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        # constant vectors
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))

        # some buffers
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.base_lin_acc = torch.zeros_like(self.base_lin_vel)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.dof_acc = torch.zeros_like(self.dof_vel)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)

        # joint targets
        self.joint_targets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, )
        self.last_joint_targets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
        self.joint_targets_rate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.joint_targets_rate_scaler = np.sqrt(self.num_actions) * self.dt

        # torque
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.total_torque = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.total_power = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # end effector
        self.num_ee = len(self.cfg.asset.ee_offsets)
        self.ee_global = torch.zeros(self.num_envs, self.num_ee, 3, dtype=torch.float, device=self.device)
        self.last_ee_global = torch.zeros(self.num_envs, self.num_ee, 3, dtype=torch.float, device=self.device)
        self.ee_local = torch.zeros(self.num_envs, self.num_ee, 3, dtype=torch.float, device=self.device)
        self.ee_contact = torch.zeros(self.num_envs, self.num_ee, dtype=torch.bool, device=self.device)
        self.ee_contact_force = torch.zeros(self.num_envs, len(self.ee_indices), dtype=torch.float, device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, len(self.ee_indices), dtype=torch.bool, device=self.device)

        self.common_step_counter = 0
        self.extras = {}
        self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]).unsqueeze(1)
        return self._get_height_at_points(points)

    def _get_height_at_points(self, points):
        """ gives the height of terrain at given points. (points in global coordinates)"""
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(points.shape[:2], device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points_c = points + self.terrain.cfg.border_size
        points_c = (points_c / self.terrain.cfg.horizontal_scale).long()
        px = points_c[:, :, 0].view(-1)
        py = points_c[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(len(points), -1) * self.terrain.cfg.vertical_scale

    def _get_mean_height_around_pos(self, pos, margin, num_points=3):
        """ gives an average height of terrain in a margin box around given position. (pos in global coordinates)
                Args: pos[num_positions, num_dims (>2)]
        """
        x = torch.linspace(-margin, margin, num_points, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, x)
        helping_points = torch.zeros(pos.shape[0], len(x) ** 2, 2, device=self.device)
        helping_points[:, :, 0] = grid_x.flatten()
        helping_points[:, :, 1] = grid_y.flatten()
        points = helping_points + pos[:, :2].unsqueeze(1).repeat(1, len(x) ** 2, 1)
        height = torch.mean(self._get_height_at_points(points), dim=1)
        return height

    def _get_max_height_around_pos(self, pos, margin):
        x = torch.tensor([-margin, 0, margin], device=self.device)
        grid_x, grid_y = torch.meshgrid(x, x)
        helping_points = torch.zeros(pos.shape[0], len(x) ** 2, 2, device=self.device)
        helping_points[:, :, 0] = grid_x.flatten()
        helping_points[:, :, 1] = grid_y.flatten()
        points = helping_points + pos[:, :2].unsqueeze(1).repeat(1, len(x) ** 2, 1)
        height = torch.amax(self._get_height_at_points(points), dim=1)
        return height

    def _compute_torques(self, joint_targets):
        # pd controller
        control_type = self.cfg.control.control_type

        if self.randomize_gains:
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        if self.cfg.domain_rand.actuator_lag:
            self.actuator_lag_buffer = torch.cat((self.actuator_lag_buffer[:, 1:, :],
                                                  joint_targets.unsqueeze(1)), dim=1)
            if self.cfg.domain_rand.randomize_actuator_lag:
                joint_targets_ = self.actuator_lag_buffer[torch.arange(self.num_envs), self.actuator_lag_index]
            else:
                joint_targets_ = self.actuator_lag_buffer[:, 0, :]
        else:
            joint_targets_ = joint_targets

        torques = self.torque_limits
        if control_type == "P":
            if self.cfg.asset.default_dof_drive_mode == 3:
                # torques = p_gains * (joint_targets_ + self.default_dof_pos - self.dof_pos - self.cfg.sim.dt * self.dof_vel)\
                #           - d_gains * (self.dof_vel - self.dof_acc * self.cfg.sim.dt) #stable pd controller (Tan et. al. 2011)
                torques = p_gains * (joint_targets_ + self.default_dof_pos - self.dof_pos) - d_gains * self.dof_vel
        elif control_type == "V":
            torques = p_gains * (joint_targets_ - self.dof_vel) - d_gains * (
                    self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = joint_targets_
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _compute_randomized_gains(self, num_envs):
        p_mult = ((
                          self.cfg.domain_rand.stiffness_multiplier_range[0] -
                          self.cfg.domain_rand.stiffness_multiplier_range[1]) *
                  torch.rand(num_envs, self.num_actions, device=self.device) +
                  self.cfg.domain_rand.stiffness_multiplier_range[1]).float()
        d_mult = ((
                          self.cfg.domain_rand.damping_multiplier_range[0] -
                          self.cfg.domain_rand.damping_multiplier_range[1]) *
                  torch.rand(num_envs, self.num_actions, device=self.device) +
                  self.cfg.domain_rand.damping_multiplier_range[1]).float()

        return p_mult * self.p_gains, d_mult * self.d_gains

    def sample_command_trajectory(self, start_point, num_points, time_horizon=1.0):
        # Generate random control points for Bézier curve
        control_points = torch.rand(num_points, 2) * 10
        control_points[0, :] = start_point

        # Create a time vector
        t = torch.arange(0, time_horizon, self.dt)

        # Calculate the Bézier curve for x and y coordinates
        x_curve = [bezier(ti, control_points[:, 0]) for ti in t]
        y_curve = [bezier(ti, control_points[:, 1]) for ti in t]

        # Calculate yaw (orientation) based on the tangent of the curve
        yaw_curve = [torch.atan2(y_curve[i + 1] - y_curve[i], x_curve[i + 1] - x_curve[i]) for i in range(len(t) - 1)]
        yaw_curve.append(yaw_curve[-1])  # Duplicate the last yaw value to match the length

        return x_curve, y_curve, yaw_curve, t

    def _prepare_reward_function(self):
        self.reward_terms = class_to_dict(self.cfg.rewards.terms)
        self.reward_num_groups = len(self.cfg.rewards.group_coeff)
        self.reward_groups = {}
        for key in self.cfg.rewards.group_coeff.keys():
            self.reward_groups[key] = []
        for name, info in self.reward_terms.items():
            group = info[0]
            self.reward_groups[group].append(name)

        self.episode_term_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self.reward_terms.keys()}
        self.episode_group_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self.reward_groups.keys()}

        self.group_rew_buf = torch.ones(self.num_envs, self.reward_num_groups, dtype=torch.float, device=self.device)
        self.rew_group_ids = self.cfg.rewards.rew_group_ids

    def compute_reward(self):
        for group_name, terms in self.reward_groups.items():
            group_idx = self.rew_group_ids[group_name]
            self.group_rew_buf[:, group_idx] = 1.0
            for i in range(len(terms)):
                reward_name = terms[i]
                reward_function = getattr(self, '_reward_' + reward_name)
                reward_config = self.reward_terms[reward_name]
                reward_sigma = reward_config[1]
                reward_tolerance = reward_config[2]
                term_reward = reward_function(reward_sigma, reward_tolerance)
                self.episode_term_sums[reward_name] += term_reward
                self.group_rew_buf[:, group_idx] *= term_reward
            self.episode_group_sums[group_name] += self.group_rew_buf[:, group_idx]

    def _draw_debug_vis(self, vis_flag='height_field'):
        """ Draws visualizations for debugging (slows down simulation a lot).
            vis_flag: options for visualization
            include 'ref_only' in vis_flag if you want to draw only for ref env

            * Note: if the viewer is disabled, the vis_flag parameters are ignored, only the quadruped will be recorded in
            the video. If the viewer is enabled, the parameters are used to configure the viewer.
        """
        self.gym.clear_lines(self.viewer)
        color_red = (1, 0, 0)
        color_green = (0, 1, 0)
        color_blue = (0, 0, 1)
        color_white = (0, 0, 0)
        color_black = (0.2, 0.2, 0.2)
        color_yellow = (0.99, 0.85, 0.05)

        if 'height_field' in vis_flag and self.terrain.cfg.measure_heights:
            # draw height lines
            # self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=color_blue)
            for i in range(self.num_envs):
                if i == self.focus_env or 'ref_only' not in vis_flag:
                    base_pos = (self.root_states[i, :3]).cpu().numpy()
                    heights = self.measured_heights[i].cpu().numpy()
                    height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                                   self.height_points[i]).cpu().numpy()
                    for j in range(heights.shape[0]):
                        x = height_points[j, 0] + base_pos[0]
                        y = height_points[j, 1] + base_pos[1]
                        z = heights[j]
                        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        return

    def draw_plane(self, env_id, x, y, z, color):
        h = 0.01
        w = 0.2
        l = 0.4
        z -= h / 2
        box_geom = gymutil.WireframeBoxGeometry(w, l, h, None, color=color)
        box_geom2 = gymutil.WireframeBoxGeometry(w / 2, l / 2, h, None, color=color)
        box_geom3 = gymutil.WireframeBoxGeometry(w * 3 / 4, l * 3 / 4, h, None, color=color)
        box_geom4 = gymutil.WireframeBoxGeometry(w / 4, l / 4, h, None, color=color)

        pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        gymutil.draw_lines(box_geom, self.gym, self.viewer, self.envs[env_id], pose)
        gymutil.draw_lines(box_geom2, self.gym, self.viewer, self.envs[env_id], pose)
        gymutil.draw_lines(box_geom3, self.gym, self.viewer, self.envs[env_id], pose)
        gymutil.draw_lines(box_geom4, self.gym, self.viewer, self.envs[env_id], pose)

    def draw_ghost_robot(self, env_id, root_state, dof_pos):
        self.dof_pos[env_id] = dof_pos
        self.dof_vel[env_id] = 0.0
        env_ids_int32 = torch.asarray([env_id], dtype=torch.int32, device=self.device)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.root_states[env_id, :7] = root_state[:7]
        self.root_states[env_id, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def get_command_idx(self, command_name):
        if isinstance(command_name, list):
            for name in command_name:
                if name in self.command_ranges:
                    return self.command_keys.index(name)
        else:
            return self.command_keys.index(command_name)

    def _set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, self.envs[self.focus_env], cam_pos, cam_target)

    def set_light(self):
        x1 = 0.4  # intensity for the first light
        y1 = 0.5  # ambient lighting for the first light
        x2 = 0.4  # intensity for the second light
        y2 = 0.5  # ambient lighting for the second light

        self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(x1, x1, x1), gymapi.Vec3(y1, y1, y1),
                                      gymapi.Vec3(0, 1, 1))
        self.gym.set_light_parameters(self.sim, 1, gymapi.Vec3(x2, x2, x2), gymapi.Vec3(y2, y2, y2),
                                      gymapi.Vec3(0, -1, 1))

    def plotter_init(self, y_getters):
        num_sub_plots = len(y_getters)
        max_data_points = 100
        self.plotter_fig, self.plotter_axs = plt.subplots(num_sub_plots)
        self.plotter_x_buffer = deque(maxlen=max_data_points)
        self.plotter_y_buffer = []
        self.plotter_lines = []
        for subplot_id in range(num_sub_plots):
            curr_plotter_ax = self.plotter_axs[subplot_id] if num_sub_plots > 1 else self.plotter_axs
            data, label, legend = y_getters[subplot_id]()
            curr_plotter_ax.set_ylabel(label)
            subplot_lines = []
            subplot_y_buffer = []
            for i in range(len(data)):
                plotter_ln, = curr_plotter_ax.plot([], [], '-')
                if len(legend) == len(data):
                    plotter_ln.set_label(legend[i])
                subplot_lines.append(plotter_ln)
                subplot_y_buffer.append(deque(maxlen=max_data_points))
            curr_plotter_ax.legend(loc='upper left')
            self.plotter_lines.append(subplot_lines)
            self.plotter_y_buffer.append(subplot_y_buffer)
        return self.plotter_lines

    def plotter_update(self, frame, x_getter, y_getters):
        if not self.is_playing:
            return self.plotter_lines

        self.plotter_x_buffer.append(x_getter())
        for (subplot_id, subplot_y_getter) in enumerate(y_getters):
            line_id = 0
            y_data, label, legend = subplot_y_getter()
            for data_value in y_data:
                self.plotter_y_buffer[subplot_id][line_id].append(data_value)
                self.plotter_lines[subplot_id][line_id].set_data(self.plotter_x_buffer,
                                                                 self.plotter_y_buffer[subplot_id][line_id])
                line_id += 1

            curr_plotter_ax = self.plotter_axs[subplot_id] if len(y_getters) > 1 else self.plotter_axs
            curr_plotter_ax.relim()
            curr_plotter_ax.autoscale_view()
        return self.plotter_lines

    def getplt_vel_fwd(self):
        label = "forward vel"
        vel_fwd = self.base_lin_vel.detach().cpu()[self.focus_env, 0]
        out = [vel_fwd]
        legend = ["actual"]
        return np.array(out), label, legend

    def getplt_vel_side(self):
        label = "side vel"
        vel_side = self.base_lin_vel.detach().cpu()[self.focus_env, 1]
        out = [vel_side]
        legend = ["actual"]
        return np.array(out), label, legend

    def getplt_ang_vel(self):
        label = "angular vel"
        ang_vel = self.base_ang_vel.detach().cpu()[self.focus_env, 2]
        out = [ang_vel]
        legend = ["actual"]
        return np.array(out), label, legend

    def getplt_base_acc_xy(self):
        base_acceleration_xy = torch.norm(self.base_lin_acc[self.focus_env, 0:2]).detach().cpu()
        label = "base_acc"
        legend = ["xy"]
        return np.array([base_acceleration_xy]), label, legend

    def getplt_joint_angles(self):
        joint_angles = self.dof_pos[self.focus_env].detach().cpu().numpy()
        joint_targets = self.joint_targets[self.focus_env].detach().cpu().numpy() + \
                        self.default_dof_pos.detach().cpu().numpy()
        return joint_angles, "joint angles", []

    def getplt_rewards(self):
        reward_terms = []
        legend = []
        for group_name, terms in self.reward_groups.items():
            for i in range(len(terms)):
                reward_name = terms[i]
                legend.append(reward_name)
                reward_function = getattr(self, '_reward_' + reward_name)
                reward_config = self.reward_terms[reward_name]
                reward_sigma = reward_config[1]
                reward_tolerance = reward_config[2]
                reward_terms.append(reward_function(reward_sigma, reward_tolerance).detach().cpu()[self.focus_env])
        label = "rewards"
        return np.array(reward_terms), label, legend
