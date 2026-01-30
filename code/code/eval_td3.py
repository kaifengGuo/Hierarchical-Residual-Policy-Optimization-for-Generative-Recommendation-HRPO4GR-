######################################
#  TD3 Evaluation on KuaiRand Env   #
######################################

import argparse
import os
import numpy as np
from tqdm import tqdm

import torch

from env import *
from model.policy import *
import utils


if __name__ == "__main__":

    # --------- 1) 先解析 env / policy class 名字 ---------
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument("--env_class", type=str, required=True,
                             help="Environment class, e.g. KRCrossSessionEnvironment_GPU")
    init_parser.add_argument("--policy_class", type=str, required=True,
                             help="Policy class, e.g. ActionTransformer")
    initial_args, _ = init_parser.parse_known_args()

    envClass = eval("{0}.{0}".format(initial_args.env_class))
    policyClass = eval("{0}.{0}".format(initial_args.policy_class))

    # --------- 2) 再解析其他参数（和 train_td3 基本一致） ---------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11, help="random seed")
    parser.add_argument("--cuda", type=int, default=-1,
                        help="cuda device number; set to -1 if using cpu")

    # 额外评估参数
    parser.add_argument("--actor_path", type=str, required=True,
                        help="Path to trained actor parameters, e.g. .../model_actor")
    parser.add_argument("--n_eval_steps", type=int, default=10000,
                        help="Total environment steps for evaluation")
    parser.add_argument("--retention_threshold", type=float, default=3.0,
                        help="A session is considered 'retained' "
                             "if return_day <= retention_threshold.")

    # 让 env / policy 自己往 parser 里加超参
    parser = envClass.parse_model_args(parser)
    parser = policyClass.parse_model_args(parser)

    args, _ = parser.parse_known_args()

    # --------- 3) 设备 & 随机种子 ---------
    if args.cuda >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        torch.cuda.set_device(args.cuda)
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    args.device = device
    utils.set_random_seed(args.seed)

    # --------- 4) 构建环境和策略 ---------
    print("Loading environment...")
    env = envClass(args)

    print("Loading policy (actor)...")
    policy = policyClass(args, env)
    policy = policy.to(device)
    policy.eval()

    # 加载训练好的 actor 参数（train_td3 保存的是 state_dict）
    print(f"Loading actor weights from: {args.actor_path}")
    actor_state = torch.load(args.actor_path, map_location=device)
    policy.load_state_dict(actor_state)

    # --------- 5) 评估循环 ---------
    print("Start evaluation...")
    observation = env.reset()

    all_return_days = []
    all_retained_flags = []

    with torch.no_grad():
        for step in tqdm(range(args.n_eval_steps)):
    
            # ----- 1) 对 ActionTransformer：直接用 observation 解包 -----
            # observation 本身就是 {'user_profile': {...}, 'user_history': {...}}
            feed_dict = {
                "user_profile": observation["user_profile"],
                "user_history": observation["user_history"],
            }
            out_dict = policy.get_forward(feed_dict)   # 直接调 get_forward
            action = out_dict["action"]                # (B, n_feedback)
    
            # ----- 2) 和之前一样，把 action 丢进 env -----
            step_dict = {"action": action}
            observation, user_feedback, _ = env.step(step_dict)
    
            # 在 KRCrossSessionEnvironment_GPU 里，只有 session 结束时
            # user_feedback["retention"] 才是 >0 的 return_day，否则为 0
            return_day = user_feedback["retention"]     # (B,)
            if return_day is not None:
                rd_np = return_day.detach().cpu().numpy()
                mask = rd_np > 0
                if mask.any():
                    valid_days = rd_np[mask]
                    all_return_days.extend(valid_days.tolist())
                    retained = (valid_days <= args.retention_threshold).astype(np.float32)
                    all_retained_flags.extend(retained.tolist())


    if len(all_return_days) == 0:
        print("Warning: no finished sessions were collected during evaluation.")
        mean_return_time = 0.0
        mean_retention = 0.0
    else:
        mean_return_time = float(np.mean(all_return_days))
        mean_retention = float(np.mean(all_retained_flags))

    print("\n========== TD3 Evaluation Result ==========")
    print(f"Total eval steps        : {args.n_eval_steps}")
    print(f"Number of finished sess.: {len(all_return_days)}")
    print(f"Return time (↓)         : {mean_return_time:.4f}")
    print(f"User retention (↑, ≤ {args.retention_threshold} days): "
          f"{mean_retention:.4f}")
    print("===========================================")
