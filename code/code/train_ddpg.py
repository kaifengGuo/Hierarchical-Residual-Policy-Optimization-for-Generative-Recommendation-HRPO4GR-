######################################
#  Code base: train_actor_critic.py  #
#  Variant:    DDPG (single critic)  #
######################################

from tqdm import tqdm
from time import time
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os

from model.agent import *
from model.policy import *
from model.critic import *
from model.buffer import *
from env import *

import utils


if __name__ == '__main__':

    # ===== 1) 先解析 class 名称（和 train_td3 一样） =====
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--env_class', type=str, required=True,
                             help='Environment class.')
    init_parser.add_argument('--policy_class', type=str, required=True,
                             help='Policy class')
    init_parser.add_argument('--critic_class', type=str, required=True,
                             help='Critic class')
    init_parser.add_argument('--agent_class', type=str, required=True,
                             help='Learning agent class (e.g., CrossSessionDDPG)')
    init_parser.add_argument('--buffer_class', type=str, required=True,
                             help='Buffer class.')

    initial_args, _ = init_parser.parse_known_args()
    print(initial_args)

    envClass = eval('{0}.{0}'.format(initial_args.env_class))
    policyClass = eval('{0}.{0}'.format(initial_args.policy_class))
    criticClass = eval('{0}.{0}'.format(initial_args.critic_class))
    agentClass = eval('{0}.{0}'.format(initial_args.agent_class))
    bufferClass = eval('{0}.{0}'.format(initial_args.buffer_class))

    # ===== 2) 通用控制参数 =====
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=11,
                        help='random seed')
    parser.add_argument('--cuda', type=int, default=-1,
                        help='cuda device number; set to -1 (default) if using cpu')

    # ===== 3) 让各个模块往 parser 里加自己的超参 =====
    parser = envClass.parse_model_args(parser)
    parser = policyClass.parse_model_args(parser)
    parser = criticClass.parse_model_args(parser)
    parser = agentClass.parse_model_args(parser)
    parser = bufferClass.parse_model_args(parser)
    args, _ = parser.parse_known_args()

    # ===== 4) 设备 & 随机种子 =====
    if args.cuda >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        torch.cuda.set_device(args.cuda)
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    args.device = device
    utils.set_random_seed(args.seed)

    # ===== 5) 构建 Environment =====
    print("Loading environment")
    env = envClass(args)

    # ===== 6) 构建 Policy (Actor) =====
    print("Setup policy:")
    policy = policyClass(args, env)
    policy.to(device)
    print(policy)

    # ===== 7) 构建单个 Critic（DDPG 用一个就够） =====
    print("Setup critic:")
    critic = criticClass(args, env, policy)
    critic = critic.to(device)
    print(critic)

    # ===== 8) 构建 Replay Buffer =====
    # buffer 接口保持和原来一致，仍然传一个“critic 列表”，只是里面只有一个
    print("Setup buffer:")
    buffer = bufferClass(args, env, policy, [critic])
    print(buffer)

    # ===== 9) 构建 Agent（CrossSessionDDPG） =====
    print("Setup agent:")
    # CrossSessionDDPG.__init__(self, args, env, actor, critic, buffer)
    agent = agentClass(args, env, policy, critic, buffer)
    print(agent)

    # ===== 10) 开始训练 =====
    try:
        print(args)
        agent.train()
    except KeyboardInterrupt:
        print("Early stop manually")
        exit_here = input("Exit completely without evaluation? (y/n) (default n):")
        if exit_here.lower().startswith('y'):
            print(os.linesep + '-' * 20 + ' END: ' + utils.get_local_time()
                  + ' ' + '-' * 20)
            exit(1)
