# created by Fatemeh Zargarbashi - 2023

from amp import ROOT_DIR
from amp.utils.helpers import get_args, get_load_path, update_cfgs_from_dict
from amp.envs import task_registry

import json
import torch
import os
import threading
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Options:
    Threading = False
    SHOW_PLOTS = False

    RSI = False
    CHANGE_MOTION_FILES = False
    Selected_Clips = [f'{ROOT_DIR}/resources/datasets/go1_motions/MANN_01_ret_go1_85_593_isaac.txt']

    EXPORT_POLICY = False

class InteractivePlay:
    """ Interactive UI to play the policy."""

    def __init__(self, args, options, full_setup=True):
        self.args = args
        self.options = options

        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

        if args.load_run is not None:
            train_cfg.runner.load_run = args.load_run
        if args.checkpoint is not None:
            train_cfg.runner.checkpoint = args.checkpoint
        load_path = get_load_path(
            os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name),
            load_run=train_cfg.runner.load_run,
            checkpoint=train_cfg.runner.checkpoint,
        )
        self.load_path = load_path
        print(f"Loading model from: {load_path}")

        # load config
        load_config_path = os.path.join(os.path.dirname(load_path), f"{train_cfg.runner.experiment_name}.json")
        with open(load_config_path) as f:
            load_config = json.load(f)
            update_cfgs_from_dict(env_cfg, train_cfg, load_config)

        # overwrite config params
        env_cfg.seed = 1
        env_cfg.env.num_envs = 1

        env_cfg.viewer.vis_flag = ['end_effector', 'ref_only']

        if 'ground_truth' in env_cfg.viewer.vis_flag:
            env_cfg.env.num_envs += 1

        if options.CHANGE_MOTION_FILES:
            env_cfg.motion_loader.motion_files = options.Selected_Clips

        env_cfg.env.play = True
        env_cfg.env.rsi_ratio = 1.0 if options.RSI else 0.0

        env_cfg.sim.use_gpu_pipeline = False
        env_cfg.motion_loader.len_preload_buf = 200  # low number for speed, but we need some for sampling the goal posture
        train_cfg.algorithm.amp_replay_buffer_size = 100

        env_cfg.viewer.overview = True

        env_cfg.observations.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_gains = False
        env_cfg.domain_rand.actuator_lag = False
        env_cfg.domain_rand.randomize_actuator_lag = False
        env_cfg.domain_rand.actuator_lag_steps = 3
        env_cfg.terrain.num_rows = 5  # reduce number of terrain levels for faster inference
        env_cfg.terrain.border_size = 5  # m

        self.env_cfg = env_cfg
        self.train_cfg = train_cfg

        if full_setup:
            # create env, runner and policy
            self.setup_all()

    def setup_all(self):
        # prepare environment
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg)
        self.env = env
        self.runner = task_registry.make_alg_runner(env=env, name=self.args.task, args=self.args, env_cfg=self.env_cfg,
                                                    train_cfg=self.train_cfg)
        self.env.reset()
        self.obs = self.env.get_observations()
        self.runner.load(self.load_path)  # load policy
        self.policy = self.runner.get_inference_policy(device=self.env.device)
        self.disc = self.runner.get_inference_disc(device=self.env.device)
        self.custom_buffer = torch.asarray([])
        self.last_dof_pos = torch.zeros_like(self.env.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.env.dof_vel)
        self.last_actions = torch.zeros_like(self.env.actions)

        # export policy as onnx model
        if self.options.EXPORT_POLICY:
            self.export_policy(self.load_path)

    def export_policy(self, load_path):
        path = os.path.join(
            os.path.dirname(load_path),
            "exported",
            "policies",
        )
        name = f'{os.path.split(os.path.dirname(load_path))[1]}'
        # export_policy_as_onnx(self.runner.policy.policy_latent_net, self.runner.policy.action_mean_net,
        #                       self.runner.actor_obs_normalizer, path, filename=f"{name}.onnx")
        print("--------------------------")
        print("Exported policy to: ", path)

    def plot_animation(self, y_getters=None):
        if y_getters is None:
            y_getters = (
                self.env.getplt_vel_fwd,
            )
            # y_getters = y_getters + self.env.getplt_all_commands()
        self.env.plotter_init(y_getters)
        self.animation = FuncAnimation(fig=self.env.plotter_fig,
                                       func=self.env.plotter_update,
                                       fargs=[self.env.get_time_stamp, y_getters],
                                       interval=100,
                                       cache_frame_data=False)
        plt.pause(0.01)

    def check_escape(self):
        for evt in self.env.gym.query_viewer_action_events(self.env.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                return True
        return False

    def play_one_step_keyframing(self):
        actions = self.policy(self.obs.detach(), self.obs_mask)
        self.env.process_keystroke(os.path.basename(os.path.dirname(self.load_path)))

        if self.env.reset_triggered:
            self.env.episode_length_buf[:] = 0
            self.env.reset_buf[:] = 1
            self.env.reset()
            self.env.reset_triggered = False

        if self.env.is_playing:
            self.obs, _, self.obs_mask, _, self.dones, _ = self.env.step(actions.detach())
        else:
            self.env.render()

    def play_one_step_amp(self):
        actions = self.policy(self.obs.detach())
        self.env.process_keystroke()

        if self.env.reset_triggered:
            self.env.episode_length_buf[:] = 0
            self.env.reset_buf[:] = 1
            self.env.reset()
            self.env.reset_triggered = False

        if self.env.is_playing:
            self.obs, _, _, _, self.dones, _ = self.env.step(actions.detach())
        else:
            self.env.render()

    def play(self):
        self.env.continuous_play = self.options.CONTINUOUS_PLAY
        if not self.env.gym.query_viewer_has_closed(self.env.viewer):
            threading.Timer(1 / 50, self.play).start()
            self.play_one_step_amp()
        else:
            self.runner.close()

    def play_with_plot(self):
        if self.options.SHOW_PLOTS:
            self.plot_animation()
        while not self.env.gym.query_viewer_has_closed(self.env.viewer):
            self.play_one_step_amp()
            if self.options.SHOW_PLOTS:
                plt.pause(0.01)
        self.runner.close()

    def reset_to_state(self, reset_state):
        env_id = torch.asarray([self.env.focus_env], device=self.env.device)
        if reset_state == "Default":
            self.env.reset()
        else:
            self.env.reset_envs_to_frames(env_id,
                                          torch.asarray(reset_state, device=self.env.device).unsqueeze(0))

    def play_manual_one_step(self, reset_state):
        actions = self.policy(self.obs.detach(), self.obs_mask)
        self.env.process_keystroke(os.path.basename(os.path.dirname(self.load_path)))

        if self.env.reset_triggered:
            self.env.episode_length_buf[:] = 0
            self.env.reset_buf[:] = 1
            self.reset_to_state(reset_state)
            self.env.reset_triggered = False

        if self.env.is_playing:
            self.obs, _, self.obs_mask, _, self.dones, _ = self.env.step(actions.detach())
        else:
            self.env.render()

    def play_manual_with_plot(self, reset_state):
        self.reset_to_state(reset_state)
        if self.options.SHOW_PLOTS:
            self.plot_animation()
        while not self.env.gym.query_viewer_has_closed(self.env.viewer):
            self.play_manual_one_step(reset_state)
            if self.options.SHOW_PLOTS:
                plt.pause(0.01)

        self.runner.close()

if __name__ == '__main__':

    args = get_args()
    options = Options()
    ip = InteractivePlay(args, options)
    ip.env.is_playing = True
    if not options.Threading:
        ip.play_with_plot()
    else:
        ip.play()
