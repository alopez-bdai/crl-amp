from amp.envs import *
from amp.utils.task_registry import task_registry
from amp.utils.helpers import get_args
from amp import ROOT_DIR


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    runner = task_registry.make_alg_runner(env=env, name=args.task, args=args, env_cfg=env_cfg)
    runner.learn()
    runner.close()


if __name__ == '__main__':
    args = get_args()
    train(args)
