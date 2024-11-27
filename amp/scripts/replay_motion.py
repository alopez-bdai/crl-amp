# created by Fatemeh Zargarbashi - 2023

from amp import ROOT_DIR
from amp.envs import *
from amp.utils.helpers import get_args
from amp.utils.task_registry import task_registry

import glob
import torch
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def replay(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)

    cfg_motion_files = env_cfg.motion_loader.motion_files
    if isinstance(cfg_motion_files, list):
        motion_files = []
        for file_name in cfg_motion_files:
            motion_files.extend(glob.glob(file_name.format(ROOT_DIR=ROOT_DIR)))
    else:
        motion_files = glob.glob(cfg_motion_files.format(ROOT_DIR=ROOT_DIR))
    motion_files = sorted(motion_files)
    num_motion_files = len(motion_files)

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, num_motion_files)
    env_cfg.sim.use_gpu_pipeline = False  # if true, then resetting the body state doesn't work
    env_cfg.env.play = True
    env_cfg.viewer.overview = False
    env_cfg.env.play = True
    env_cfg.motion_loader.len_preload_buf = num_motion_files
    env_cfg.viewer.vis_flag = ['end_effector']
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.motion_loader.amp_len_preload_buf = 10

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    traj_idxs = np.arange(env.num_envs)

    # reset to 1st frame
    i = 0  # frame number
    times = 0 * env.amp_loader.trajectory_frame_dt[traj_idxs]
    reference_frames = env.amp_loader.get_full_frame_at_time_batch(traj_idxs=traj_idxs, times=times)
    env.reset_envs_to_frames(torch.arange(env.num_envs, device=env.device), reference_frames)

    # simulate an episode
    # y_getters = (env.getplt_joint_angles,)
    # env.plotter_init(y_getters)
    # FuncAnimation(fig=env.plotter_fig,
    #               func=env.plotter_update,
    #               fargs=[env.get_time_stamp, y_getters],
    #               interval=100,
    #               cache_frame_data=False)
    # plt.pause(0.01)
    while True:
        env.process_keystroke()
        if env.reset_triggered:
            env.play_step = 0
            env.reset_triggered = False

        if env.is_playing:
            times = np.minimum(env.play_step / SLOW_MOTION_RATE * env.amp_loader.trajectory_frame_dt[traj_idxs],
                               env.amp_loader.trajectory_lens[traj_idxs] - env.amp_loader.trajectory_frame_dt[
                                   traj_idxs])
            reference_frames = env.amp_loader.get_full_frame_at_time_batch(traj_idxs=traj_idxs, times=times)
            # play directly from frames:
            # frame_id = np.minimum(env.play_step, env.amp_loader.trajectory_num_frames - 1).astype(int)
            # reference_frames = torch.vstack([env.amp_loader.trajectories_full[traj_id][frame_id[traj_id]] for traj_id in traj_idxs])
            env.reset_envs_to_frames(torch.arange(env.num_envs, device=env.device), reference_frames)
            env.play_step += 1
        env.render(replay=True)
        env.common_step_counter += 1
        plt.pause(0.01)


if __name__ == '__main__':
    SLOW_MOTION_RATE = 1
    args = get_args()
    replay(args)
