from tqdm import tqdm
from time import time
import torch
import argparse
import numpy as np
import os
import random

from model.agent import *
from model.policy import *
from model.critic import *
from model.buffer import *
from env import KREnvironment_WholeSession_GPU

import utils


def parse_args():
    """
    1) 先用一个简易 parser 拿到各个 class 的名字（字符串）；
    2) 用 env_class / policy_class / critic_class / agent_class / buffer_class 决定具体的类对象（module.Class）；
    3) 再让这些类往 parser 里面加自己的超参。
    """
    # 初次解析，只关心类名和少量全局参数
    init_parser = argparse.ArgumentParser(add_help=False)
    init_parser.add_argument('--env_class', type=str, required=True, help='Environment class.')
    init_parser.add_argument('--policy_class', type=str, required=True, help='Policy class')
    init_parser.add_argument('--critic_class', type=str, required=True, help='Critic class')
    init_parser.add_argument('--agent_class', type=str, required=True, help='Learning agent class')
    init_parser.add_argument('--buffer_class', type=str, required=True, help='Buffer class.')

    init_parser.add_argument('--seed', type=int, default=11, help='random seed')
    init_parser.add_argument('--cuda', type=int, default=-1, help='cuda device; set -1 for cpu')

    # 评估专用参数（默认采用 episode-wise 口径，对齐 eval_onerec_value_rerank.py）
    # - num_episodes: 收集多少个“完成的 episode（done=用户离开）”后停止
    # - eval_episodes: 旧参数名，保留兼容（现在默认也视为 num_episodes）
    init_parser.add_argument('--num_episodes', type=int, default=None,
                             help='number of finished episodes to aggregate (episode-wise). '
                                  'If set, overrides --eval_episodes.')
    init_parser.add_argument('--eval_episodes', type=int, default=200,
                             help='[compat] alias of --num_episodes (episode-wise)')
    init_parser.add_argument('--eval_epsilon', type=float, default=0.0,
                             help='epsilon for evaluation policy')
    init_parser.add_argument('--log_every', type=int, default=None,
                             help='log every N finished episodes. If not set, use --log_interval.')
    init_parser.add_argument('--log_interval', type=int, default=50,
                             help='[compat] log frequency; used as fallback for --log_every')

    initial_args, _ = init_parser.parse_known_args()

    # ★ 关键：和 train_actor_critic.py 保持一致，拿到“类对象”而不是模块
    # env.KREnvironment_WholeSession_GPU.KREnvironment_WholeSession_GPU
    envClass    = eval('{0}.{0}'.format(initial_args.env_class))
    policyClass = eval('{0}.{0}'.format(initial_args.policy_class))
    criticClass = eval('{0}.{0}'.format(initial_args.critic_class))
    agentClass  = eval('{0}.{0}'.format(initial_args.agent_class))
    bufferClass = eval('{0}.{0}'.format(initial_args.buffer_class))

    # 第二轮 parser，真正收集所有超参
    parser = argparse.ArgumentParser()

    # 把刚才的“控制类参数”也写进来（带默认值）
    parser.add_argument('--env_class', type=str, default=initial_args.env_class)
    parser.add_argument('--policy_class', type=str, default=initial_args.policy_class)
    parser.add_argument('--critic_class', type=str, default=initial_args.critic_class)
    parser.add_argument('--agent_class', type=str, default=initial_args.agent_class)
    parser.add_argument('--buffer_class', type=str, default=initial_args.buffer_class)

    parser.add_argument('--seed', type=int, default=initial_args.seed)
    parser.add_argument('--cuda', type=int, default=initial_args.cuda)
    parser.add_argument('--num_episodes', type=int, default=initial_args.num_episodes)
    parser.add_argument('--eval_episodes', type=int, default=initial_args.eval_episodes)
    parser.add_argument('--eval_epsilon', type=float, default=initial_args.eval_epsilon)
    parser.add_argument('--log_every', type=int, default=initial_args.log_every)
    parser.add_argument('--log_interval', type=int, default=initial_args.log_interval)

    # 让各个模块把自己的超参数塞进来（和 train_actor_critic 完全同源）
    parser = envClass.parse_model_args(parser)
    parser = policyClass.parse_model_args(parser)
    parser = criticClass.parse_model_args(parser)
    parser = agentClass.parse_model_args(parser)
    parser = bufferClass.parse_model_args(parser)

    args = parser.parse_args()

    return args, envClass, policyClass, criticClass, agentClass, bufferClass


def build_agent_and_env(args, envClass, policyClass, criticClass, agentClass, bufferClass):
    """
    完全照抄 train_actor_critic.py 的构建顺序：
      env -> policy -> critic -> buffer -> agent
    """
    # device 设置也和 train 一样，用字符串 'cuda:0' / 'cpu'
    if args.cuda >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        torch.cuda.set_device(args.cuda)
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    args.device = device
    utils.set_random_seed(args.seed)

    # Environment
    print("Loading environment")
    env = envClass(args)

    # Policy, Critic, Buffer, Agent
    print("Setup policy:")
    policy = policyClass(args, env)
    policy.to(device)
    print(policy)

    print("Setup critic:")
    if args.agent_class == 'TD3':
        critic1 = criticClass(args, env, policy)
        critic1.to(device)
        critic2 = criticClass(args, env, policy)
        critic2.to(device)
        critic = [critic1, critic2]
    else:
        critic = criticClass(args, env, policy)
        critic.to(device)
    print(critic)

    print("Setup buffer:")
    buffer = bufferClass(args, env, policy, critic)
    print(buffer)

    print("Setup agent:")
    agent = agentClass(args, env, policy, critic, buffer)
    print(agent)

    # 保底一下 device
    if not hasattr(agent, "device"):
        agent.device = device

    return env, agent


def _load_actor_only(agent, args):
    """
    只加载 actor（critic 维度不匹配就算了），以防止因为你改过 critic 结构导致整个 load 失败。
    """
    if not hasattr(args, "save_path") or args.save_path is None or args.save_path == "":
        print("[WARN] args.save_path 未设置，无法加载 actor checkpoint")
        return

    actor_path = args.save_path + "_actor"
    if not os.path.exists(actor_path):
        print(f"[WARN] 找不到 actor checkpoint: {actor_path}")
        return

    print(f"[Fallback] 只加载 actor 参数: {actor_path}")
    state = torch.load(actor_path, map_location=agent.device)
    model_dict = agent.actor.state_dict()

    # 只加载形状对得上的参数，防止你后面又改了网络层数
    filtered = {}
    for k, v in state.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered[k] = v
    model_dict.update(filtered)
    agent.actor.load_state_dict(model_dict)

    if hasattr(agent, "actor_target"):
        import copy
        agent.actor_target = copy.deepcopy(agent.actor)

    missing = []
    shape_bad = []
    for k, v in state.items():
        if k not in model_dict:
            missing.append(k)
        elif model_dict[k].shape != v.shape:
            shape_bad.append((k, tuple(v.shape), tuple(model_dict[k].shape)))
    
    print("[Dbg] missing-in-model (ckpt has, model not):", len(missing))
    print("  ", missing[:30])
    
    print("[Dbg] shape-mismatch:", len(shape_bad))
    print("  ", shape_bad[:30])




def compute_step_reward(env, response_dict):
    """
    根据 immediate_response 和 env.response_weights 计算一步的 reward：
      reward(b) = sum_{slot, feedback} response[b,slot,f] * w[f]
    """
    resp = response_dict["immediate_response"]  # (B, slate, n_feedback)
    if isinstance(resp, np.ndarray):
        resp = torch.from_numpy(resp)

    if hasattr(env, "response_weights"):
        w = env.response_weights  # (n_feedback,)
        if isinstance(w, np.ndarray):
            w = torch.from_numpy(w)
        w = w.view(1, 1, -1).to(resp.device)
        reward = (resp * w).sum(dim=2).sum(dim=1)  # (B,)
    else:
        # 保底：只看第一个反馈通道
        reward = resp[..., 0].sum(dim=1)

    return reward  # (B,)


def evaluate_agent(agent, env, args):
    """
    评估逻辑（不走 facade，直接 env.reset / env.step + agent.apply_policy）：

      1. 先尝试 agent.load() 完整加载；失败则只加载 actor。
      2. 用 env.reset 拉一批用户（episode_batch_size）。
      3. 每一步：
           - policy_output = agent.apply_policy(obs, agent.actor, epsilon, do_explore=False, do_softmax=True)
           - indices = policy_output["indices"]
           - next_obs, response_dict, update_info = env.step({"action": indices})
           - 从 response_dict["immediate_response"] + env.response_weights 算 reward。
      4. 输出：
           - avg_step_reward
           - discounted_return（用 gamma 衰减）
           - env.get_report()（平均步长 / coverage / ILD / leave 等）
    """
    print("========== Loading trained parameters ==========")
    loaded_ok = False
    if hasattr(agent, "load"):
        try:
            agent.load()
            loaded_ok = True
            print("[Info] 成功通过 agent.load() 加载完整模型")
        except Exception as e:
            print("[Warn] agent.load() 失败，将退化为只加载 actor：")
            print("       ", repr(e))

    if not loaded_ok:
        _load_actor_only(agent, args)

    # eval 模式
    agent.actor.eval()
    if hasattr(agent, "critic") and hasattr(agent.critic, "eval"):
        agent.critic.eval()

    # 初始化环境
    if not hasattr(args, "episode_batch_size"):
        raise ValueError("args.episode_batch_size 未设置，请在 .sh 里传 --episode_batch_size")

    obs = env.reset({"batch_size": args.episode_batch_size})

    # ================================
    # Episode-wise evaluation (对齐 rerank)
    # ================================
    # 兼容旧参数名：--eval_episodes 以前表示“环境步数”，现在默认视为“完成 episode 数”
    # 你也可以在 .sh 里显式传 --num_episodes 来避免歧义。
    num_episodes = getattr(args, "num_episodes", None)
    if num_episodes is None:
        num_episodes = getattr(args, "eval_episodes", 200)

    log_every = getattr(args, "log_every", None)
    if log_every is None:
        # 兼容旧参数：log_interval
        log_every = getattr(args, "log_interval", 50)

    epsilon = getattr(args, "eval_epsilon", 0.0)
    B = int(args.episode_batch_size)
    device = agent.device if hasattr(agent, "device") else args.device

    cur_returns = torch.zeros(B, device=device)
    cur_lengths = torch.zeros(B, device=device)
    finished = 0
    all_ret, all_len = [], []

    # ================================
    # Behavior rate statistics (episode-wise)
    #   - numerator: behavior count (sum over items)
    #   - denominator: impressions = shown items (steps * slate_size)
    # Only accumulate finished episodes to align with total_reward/depth.
    # ================================
    beh_names = None
    if hasattr(env, "response_types"):
        try:
            beh_names = list(env.response_types)
        except Exception:
            beh_names = None
    if beh_names is None:
        try:
            beh_names = env.reader.get_statistics().get("feedback_type", None)
        except Exception:
            beh_names = None
    if beh_names is None and hasattr(env, "response_weights"):
        try:
            K_tmp = int(env.response_weights.shape[0])
            beh_names = [f"fb{i}" for i in range(K_tmp)]
        except Exception:
            beh_names = None
    if beh_names is None:
        beh_names = ["is_click", "long_view", "is_like", "is_comment", "is_forward", "is_follow", "is_hate"]

    K = len(beh_names)
    cur_beh_counts = torch.zeros(B, K, device=device)   # (B, K)
    cur_impr = torch.zeros(B, device=device)            # (B,)
    total_beh_counts = torch.zeros(K, device=device)    # (K,)
    total_impr = 0.0

    print(
        f"========== Start evaluation: num_episodes={num_episodes}, "
        f"batch={B}, epsilon={epsilon} =========="
    )
    t_start = time()

    def _ensure_action_shape(x, slate_size: int):
        """env.step 期望 action shape=(B, slate_size)."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.to(device)
        if x.dim() == 1:
            x = x.view(-1, 1)
        if x.size(1) != slate_size:
            # 常见情况：策略只吐出 (B,) 或 (B,1)，这里尽量自动修复。
            if x.size(1) == 1 and slate_size > 1:
                x = x.repeat(1, slate_size)
        return x

    with torch.no_grad():
        # 直到收集够 num_episodes 个 finished episodes
        # 环境会在 done 后自动替换用户并清零 temper/step，所以我们只需手动清零统计量。
        while finished < num_episodes:
            if args.agent_class in ["DDPG", "HAC"]:
                policy_output = agent.apply_policy(
                    obs,
                    agent.actor,
                    epsilon,  # exploration 强度，eval 通常 0
                    False,    # do_explore = False
                    False,    # is_train = False
                )
            else:
                policy_output = agent.apply_policy(
                    obs,
                    agent.actor,
                    epsilon=epsilon,
                    do_explore=False,
                    do_softmax=True,
                )

            if "indices" not in policy_output:
                raise RuntimeError("policy_output 里没有 'indices' 字段，请确认策略输出包含 indices")

            indices = _ensure_action_shape(policy_output["indices"], int(args.slate_size))

            next_obs, response_dict, _ = env.step({"action": indices})

            # Behavior stats update for this step
            im = response_dict.get("immediate_response", None)
            if im is None:
                raise RuntimeError("response_dict 里没有 'immediate_response' 字段，无法统计行为比率")
            if isinstance(im, np.ndarray):
                im = torch.from_numpy(im)
            im = im.to(device).float()  # (B, slate, K)

            # 每步曝光 = slate_size（每个用户每步展示多少 item）
            cur_impr += float(im.size(1))

            # 行为发生次数：对 slate 维求和 -> (B, K)
            # 如果 K 不匹配（例如 beh_names 兜底不准），就按 min(K, im.size(2)) 取交集
            K_eff = min(K, int(im.size(2)))
            if K_eff < K:
                cur_beh_counts[:, :K_eff] += im[:, :, :K_eff].sum(dim=1)
            else:
                cur_beh_counts += im.sum(dim=1)

            # per-user step reward: (B,)
            step_r = compute_step_reward(env, response_dict).to(device)
            cur_returns += step_r
            cur_lengths += 1

            done = response_dict.get("done", None)
            if done is None:
                raise RuntimeError("response_dict 里没有 'done' 字段，无法做 episode-wise 统计")
            if isinstance(done, np.ndarray):
                done = torch.from_numpy(done)
            done = done.to(device).bool()

            if done.any():
                idxs = torch.nonzero(done, as_tuple=False).squeeze(-1)
                for idx in idxs.tolist():
                    if finished < num_episodes:
                        all_ret.append(float(cur_returns[idx].item()))
                        all_len.append(float(cur_lengths[idx].item()))
                        finished += 1
                        total_beh_counts += cur_beh_counts[idx]
                        total_impr += float(cur_impr[idx].item())
                        if (finished % log_every) == 0:
                            print(
                                f"Progress {finished}/{num_episodes} | "
                                f"Ret: {float(np.mean(all_ret)):.4f} | "
                                f"Len: {float(np.mean(all_len)):.2f}"
                            )

                # 清零已结束用户的累计量（环境会自动替换这些用户）
                cur_returns[done] = 0
                cur_lengths[done] = 0
                cur_beh_counts[done] = 0
                cur_impr[done] = 0

            obs = next_obs

    t_end = time()
    print(f"========== Evaluation finished in {t_end - t_start:.2f} seconds ==========")

    # 汇总（对齐 rerank 输出）
    if all_ret:
        total_ret = float(np.mean(all_ret))
        depth = float(np.mean(all_len))
        avg_step_r = total_ret / depth if depth > 0 else 0.0

        print("=" * 40)
        print(f"Agent: {args.agent_class}")
        print(f"Total Reward: {total_ret:.4f}")
        print(f"Depth: {depth:.2f}")
        print(f"Avg Step Reward: {avg_step_r:.4f}")

        if hasattr(env, "get_report"):
            env_report = env.get_report()
            coverage = float(env_report.get("coverage", 0.0))
            ild = float(env_report.get("ILD", 0.0))
            print(f"Coverage: {coverage:.2f}")
            print(f"ILD: {ild:.4f}")

            # Behavior rates
            print("Behavior rates (count / impressions):")
            if total_impr <= 0:
                print("  [WARN] total_impr=0")
            else:
                for k, name in enumerate(beh_names):
                    cnt = float(total_beh_counts[k].item())
                    rate = 100.0 * cnt / total_impr
                    print(f"  {name}: {int(round(cnt))}/{int(round(total_impr))} ({rate:.4f}%)")

            print("Table-style metrics:")
            print(f"Depth: {depth:.2f}")
            print(f"Average reward: {avg_step_r:.4f}")
            print(f"Total reward: {total_ret:.4f}")
            print(f"Coverage: {coverage:.2f}")
            print(f"ILD: {ild:.4f}")
        print("=" * 40)

        return {
            "num_episodes": int(num_episodes),
            "total_reward_mean": total_ret,
            "depth_mean": depth,
            "avg_step_reward": avg_step_r,
        }

    print("[WARN] all_ret 为空：可能 eval 过程中没有触发 done（用户未离开）")
    return {
        "num_episodes": int(num_episodes),
        "total_reward_mean": 0.0,
        "depth_mean": 0.0,
        "avg_step_reward": 0.0,
    }


def main():
    args, envClass, policyClass, criticClass, agentClass, bufferClass = parse_args()
    print("========== Parsed arguments (partial) ==========")
    print(f"env_class={args.env_class}, policy_class={args.policy_class}, "
          f"critic_class={args.critic_class}, agent_class={args.agent_class}, "
          f"buffer_class={args.buffer_class}")

    env, agent = build_agent_and_env(args, envClass, policyClass, criticClass, agentClass, bufferClass)

    # 评估
    evaluate_agent(agent, env, args)


if __name__ == "__main__":
    main()
