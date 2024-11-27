from amp.cfg.amp.amp_config import AMPCfg, AMPTrainCfg


class Go1AMPCfg(AMPCfg):
    class env(AMPCfg.env):
        num_envs = 4096

        num_actions = 12
        num_observations = 49

        reference_state_initialization = True  # initialize state from reference data
        rsi_ratio = 0.8

    class motion_loader(AMPCfg.motion_loader):
        preload_mode = 'transition'  # 'transition' or 'trajectory'
        motion_files = '{ROOT_DIR}/resources/datasets/go1_motions/*.txt'
        special_motion_weight = 3.0
        num_amp_frames = 2
        # len_preload_buf = 4096  # if preload trajectories
        len_preload_buf = 200000  # if preload transitions
        nominal_height = 0.32  # [m]


    class init_state(AMPCfg.init_state):
        pos = [0.0, 0.0, 0.32]  # x,y,z [m]

        # default joint angles =  target angles [rad] when action = 0.0
        default_joint_angles = {
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control(AMPCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        stiffness = {'joint': 25.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]


    class asset(AMPCfg.asset):
        file = '{ROOT_DIR}/resources/robots/go1/go1.urdf'
        foot_name = "calf"
        penalize_contacts_on = ["thigh"]
        terminate_after_contacts_on = ["base", "thigh"]
        terminate_on_joint_limit = True
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        ee_offsets = {
            "FL_calf": [0.0, 0.0, -0.213],
            "FR_calf": [0.0, 0.0, -0.213],
            "RL_calf": [0.0, 0.0, -0.213],
            "RR_calf": [0.0, 0.0, -0.213],
        }

    class commands(AMPCfg.commands):
        class ranges:
            lin_vel_x = [-0.5, 2.5]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]  # min max [m/s]
            ang_vel_z = [-1.57, 1.57]  # min max [rad/s]

    class rewards(AMPCfg.rewards):
        class terms:
            # group, sigma, tolerance
            joint_targets_rate = ['reg', 10.0, 0.0]
            joint_soft_limits = ['reg', 0.1, 0.0]
            # stand_still = [0, 0.2, 0.1]
            # feet_height = [0, 0.1, 0.0]
            lin_vel_x = ['task', 0.3, 0.05]
            lin_vel_y = ['task', 0.3, 0.05]
            ang_vel_z = ['task', 0.6, 0.05]

        group_coeff = {'reg': 0.1, 'task': 0.3}

        group_coeff_curriculum = {'reg': False, 'task': False}
        curr_start_iter = 1000
        curr_end_iter = 1001
        change_value = {'reg': 0, 'task': 6.0}

        feet_height_target = 0.07  # [m]

    class terrain(AMPCfg.terrain):
        mesh_type = 'plane'  # trimesh
        measure_heights = False
        curriculum = False

        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping_stone, gap, pit, wall]
        terrain_proportions = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        vertical_scale = 0.001  # [m]
        border_size = 4  # [m]
        max_init_terrain_level = 1  # starting curriculum state
        num_rows = 16  # number of terrain rows (levels)
        num_cols = 8  # number of terrain cols (types)
        randomize_robot_origins = True  # only effective is mesh_type is not 'plane

    class domain_rand(AMPCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.25, 1.5]

        randomize_base_mass = False
        added_mass_range = [-1., 1.]

        push_robots = True
        push_interval_s = 2
        max_push_vel_xyz = 1.0
        max_push_avel_xyz = 1.0

        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

        actuator_lag = False
        randomize_actuator_lag = False
        actuator_lag_steps = 6  # the lag simulated would be actuator_lag_steps * dt / decimation



class Go1AMPTrainCfg(AMPTrainCfg):
    algorithm_name = 'AMP'

    class runner(AMPTrainCfg.runner):
        run_name = 'vel'
        experiment_name = 'go1_amp'
        max_iterations = 20000  # number of policy updates

    class algorithm(AMPTrainCfg.algorithm):
        learning_rate = 1.e-4
        schedule = 'fixed'
        entropy_coef = 0.01
        disc_learning_rate = 1.e-4
        disc_loss = "LSGAN"  # "WGAN" or "LSGAN"
        amp_coef = 0.6  # loss coefficient multiplied by amp loss for multi-critic
        surrogate_coef = 10
        bootstrap = True
        normalize_coeffs = False

    class policy(AMPTrainCfg.policy):
        log_std_init = 0.0
        actor_hidden_dims = [512, 128]
        critic_hidden_dims = [512, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
