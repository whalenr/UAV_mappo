"""
# @Time    : 2021/6/30 10:07 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : train.py
"""

# !/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from config import get_config
from environment2.env_wrappers import SubprocVecEnv, DummyVecEnv
from environment2.Constant import N_ETUAV, N_DPUAV


def make_train_env(all_args):
    def get_env_fn():
        def init_env():
            from environment2.Area import Area
            env = Area()
            return env

        return init_env

    return DummyVecEnv([get_env_fn() for _ in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn():
        def init_env():
            from environment2.Area import Area
            env = Area()
            return env

        return init_env

    return DummyVecEnv([get_env_fn() for _ in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--num_agents', type=int, default=N_ETUAV+N_DPUAV, help="number of players")
    """定义agent数量"""

    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    # 定义数据
    parser = get_config()
    all_args = parse_args(args, parser)

    # # 判断使用卷积网络是否正确
    # if all_args.algorithm_name == "rmappo":
    #     assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    # elif all_args.algorithm_name == "mappo":
    #     assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
    #         "check recurrent policy!")
    # else:
    #     raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 定义数据存放路径
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 设置进程名
    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.experiment_name) + "@" + str(
        all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # 生成环境
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    """定义用户个数"""
    num_agents = N_ETUAV + N_DPUAV

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # 定义runner
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner
    runner = Runner(config)

    # 主程序入口
    runner.run()

    # ？？？
    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
