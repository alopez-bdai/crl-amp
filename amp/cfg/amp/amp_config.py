from amp.cfg.base.base_config import BaseEnvCfg, BaseTrainCfg


class AMPCfg(BaseEnvCfg):
    seed = 42

    class env(BaseEnvCfg.env):
        num_envs = 4096

        include_history_steps = None  # Number of steps of history to include.

        num_observations = 1  # should be overwritten
        num_privileged_obs = None  # None
        num_actions = 1  # should be overwritten

        episode_length = 250  # episode length

        reference_state_initialization = True  # initialize state from reference data

        play = False
        debug = False

    class motion_loader:
        motion_files = ''

        preload_mode = 'trajectory'  # 'transition' or 'trajectory'
        # len_preload_buf = 2000000  # if preload transitions
        len_preload_buf = 4096  # if preload trajectories
        max_len_preload_trajs = 256  # max num time steps for splitting trajs

    class init_state(BaseEnvCfg.init_state):
        pos = [0.0, 0.0, 1.0]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # default joint angles =  target angles [rad] when action = 0.0
        default_joint_angles = None  # should be overwritten

    class control(BaseEnvCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        scale_joint_target = 0.25
        clip_joint_target = 100.
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class commands(BaseEnvCfg.commands):
        # default: lin_vel_x, lin_vel_y, ang_vel_z
        change_commands = False
        change_commands_interval_s = 10.  # time before command are changed[s]

        # keyboard incrementation value
        keyboard_increment = [0.1]

        class ranges:
            lin_vel_x = [-1.0, 2.0]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]  # min max [m/s]
            ang_vel_z = [-1.57, 1.57]  # min max [rad/s]

    class rewards(BaseEnvCfg.rewards):
        class terms:
            # group, sigma, tolerance
            joint_targets_rate = ['reg', 10.0, 0.0]
            lin_vel_x = ['task', 0.3, 0.1]
            lin_vel_y = ['task', 0.3, 0.0]
            ang_vel_z = ['task', 0.6, 0.0]

        rew_group_ids = {'reg': 0, 'task': 1}  # this is based on algorithm implementation

        group_coeff = {'reg': 0.1, 'task': 0.4}

        group_coeff_curriculum = {'reg': False, 'task': False}
        curr_start_iter = 1000
        curr_end_iter = 1001
        change_value = {'reg': 0, 'task': 0.0}

        only_positive_rewards = True  # if true negative total rewards are clipped at zero
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.

    class domain_rand(BaseEnvCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.25, 1.75]

        randomize_base_mass = False
        added_mass_range = [-1., 1.]

        push_robots = False
        push_interval_s = 15
        max_push_vel_xyz = 1.0
        max_push_avel_xyz = 1.0

        randomize_gains = False
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

        actuator_lag = False
        randomize_actuator_lag = False
        actuator_lag_steps = 3  # the lag simulated would be actuator_lag_steps * dt / decimation

    class observations:
        clip_obs = True
        clip_limit = 100.
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05

    class termination:
        max_base_lin_vel = 100.0
        max_base_ang_vel = 1000.0
        max_rel_height = 3.0

    class viewer(BaseEnvCfg.viewer):
        enable_viewer = False
        overview = True
        ref_pos_b = [1.5, 1.5, 0.5]
        record_camera_imgs = True
        camera_pos_b = [0.65, 0.65, 0.5]
        vis_flag = ['ref_only']


class AMPTrainCfg(BaseTrainCfg):
    algorithm_name = 'AMP'  # 'AMP' or 'AMPMultiCritic'

    class policy:
        log_std_init = 0.0
        actor_hidden_dims = [256, 128]
        critic_hidden_dims = [256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs * num_steps / num_minibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

        amp_replay_buffer_size = 1000000
        amp_coef = 2.0  # loss coefficient multiplied by amp loss
        disc_hidden_dims = [256, 128]
        disc_learning_rate = 1e-5
        disc_loss = "LSGAN"  # "WGAN" or "LSGAN"
        normalize_coeffs = True

    class runner:
        num_steps_per_env = 24  # per iteration
        max_iterations = 2000  # number of policy updates
        normalize_observation = True
        normalize_disc_input = True
        save_interval = 100  # check for potential saves every this many iterations

        record_gif = True  # need to enable env.viewer.record_camera_imgs and run with wandb
        record_gif_interval = 50
        record_iters = 10  # should be int * num_st   eps_per_env

        # logging
        run_name = 'test'
        experiment_name = 'amp'
        entity = "fatemehzargar"

        # load and resume
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model

        wandb = True
        wandb_group = "default"
