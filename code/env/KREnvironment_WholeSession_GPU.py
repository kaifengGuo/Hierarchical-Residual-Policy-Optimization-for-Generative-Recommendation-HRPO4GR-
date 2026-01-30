import numpy as np
import torch
import random
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import torch.nn.functional as F

import utils
from reader import KRMBSeqReader
from model.simulator import *
from env.BaseRLEnvironment import BaseRLEnvironment

def neg_sample_calibrate_probs(p, s, eps=1e-8):
    """
    p:  (..., n_feedback)  in [0,1]  (URM predicted under neg down-sampling / neg re-weight)
    s:  (..., n_feedback)  keep probability for negatives (or pos/neg ratio), typically small for rare events
    return: p_real in [0,1]
    
    Formula:
      odds_real = s * odds_model
      p_real = (s*p) / ((1-p) + s*p)
    """
    p = p.clamp(eps, 1.0 - eps)
    s = s.clamp(min=eps)
    return (s * p) / ((1.0 - p) + s * p)


class KREnvironment_WholeSession_GPU(BaseRLEnvironment):
    '''
    KuaiRand simulated environment for consecutive list-wise recommendation
    Main interface:
    - parse_model_args: for hyperparameters
    - reset: reset online environment, monitor, and corresponding initial observation
    - step: action --> new observation, user feedbacks, and other updated information
    - get_candidate_info: obtain the entire item candidate pool
    Main Components:
    - data reader: self.reader for user profile&history sampler
    - user immediate response model: see self.get_response
    - no user leave model: see self.get_leave_signal
    - candidate item pool: self.candidate_ids, self.candidate_item_meta
    - history monitor: self.env_history, not set up until self.reset
    '''
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - uirm_log_path
        - slate_size
        - episode_batch_size
        - item_correlation
        - single_response
        - from BaseRLEnvironment
            - max_step_per_episode
            - initial_temper
        '''
        parser = BaseRLEnvironment.parse_model_args(parser)
        parser.add_argument('--uirm_log_path', type=str, required=True, 
                            help='log path for pretrained user immediate response model')
        parser.add_argument('--slate_size', type=int, required=6, 
                            help='number of item per recommendation slate')
        parser.add_argument('--episode_batch_size', type=int, default=32, 
                            help='episode sample batch size')
        parser.add_argument('--item_correlation', type=float, default=0, 
                            help='magnitude of item correlation')
        parser.add_argument('--single_response', action='store_true', 
                            help='only include the first feedback as reward signal')
        return parser
    
    def __init__(self, args):
        '''
        from BaseRLEnvironment:
            self.max_step_per_episode
            self.initial_temper
        self.uirm_log_path
        self.slate_size
        self.rho
        self.immediate_response_stats: reader statistics for user response model
        self.immediate_response_model: the ground truth user response model
        self.max_hist_len
        self.response_types
        self.response_dim: number of feedback_type
        self.response_weights
        self.reader
        self.candidate_iids: [encoded item id]
        self.candidate_item_meta: {'if_{feature_name}': (n_item, feature_dim)}
        self.n_candidate
        self.candidate_item_encoding: (n_item, item_enc_dim)
        self.gt_state_dim: ground truth user state vector dimension
        self.action_dim: slate size
        self.observation_space: see reader.get_statistics()
        self.action_space: n_condidate
        '''
        super(KREnvironment_WholeSession_GPU, self).__init__(args)
        self.uirm_log_path = args.uirm_log_path
        self.slate_size = args.slate_size
        self.episode_batch_size = args.episode_batch_size
        self.rho = args.item_correlation
        self.single_response = args.single_response
        
        infile = open(args.uirm_log_path, 'r')
        class_args = eval(infile.readline()) # example: Namespace(model='RL4RSUserResponse', reader='RL4RSDataReader')
        model_args = eval(infile.readline()) # model parameters in Namespace
        print("Environment arguments: \n" + str(model_args))
        infile.close()
        print("Loading raw data")
        assert (class_args.reader == 'KRMBSeqReader' or class_args.reader == 'MLSeqReader') and 'KRMBUserResponse' in class_args.model
        
        print("Load user sequence reader")
        reader, reader_args = self.get_reader(args.uirm_log_path) # definition in base
        self.reader = reader
        print(self.reader.get_statistics())
        
        print("Load immediate user response model")
        uirm_stats, uirm_model, uirm_args = self.get_user_model(args.uirm_log_path, args.device) # definition in base
        self.immediate_response_stats = uirm_stats
        self.immediate_response_model = uirm_model
        self.max_hist_len = uirm_stats['max_seq_len']
        self.response_types = uirm_stats['feedback_type']
        self.response_dim = len(self.response_types)
        self.response_weights = torch.tensor(list(self.reader.get_response_weights().values())).to(torch.float).to(args.device)
        if args.single_response:
            self.response_weights = torch.zeros_like(self.response_weights)
            self.response_weights[0] = 1
        
        
        print("Setup candidate item pool")
        
        # [encoded item id], size (n_item,)
        self.candidate_iids = torch.tensor([reader.item_id_vocab[iid] for iid in reader.items]).to(self.device)
        
        # item meta: {'if_{feature_name}': (n_item, feature_dim)}
        candidate_meta = [reader.get_item_meta_data(iid) for iid in reader.items]
        self.candidate_item_meta = {}
        self.n_candidate = len(candidate_meta)
        for k in candidate_meta[0]:
            self.candidate_item_meta[k] = torch.FloatTensor(np.concatenate([meta[k] for meta in candidate_meta]))\
                                                    .view(self.n_candidate,-1).to(self.device)
            
        # (n_item, item_enc_dim), groud truth encoding is implicit to RL agent
        item_enc, _ = self.immediate_response_model.get_item_encoding(self.candidate_iids, 
                                               {k[3:]:v for k,v in self.candidate_item_meta.items()}, 1)
        self.candidate_item_encoding = item_enc.view(-1,self.immediate_response_model.enc_dim)
        
        # spaces
        self.gt_state_dim = self.immediate_response_model.state_dim
        self.action_dim = self.slate_size
        self.observation_space = self.reader.get_statistics()
        self.action_space = self.n_candidate
        
        self.immediate_response_model.to(args.device)
        self.immediate_response_model.device = args.device
        
        
    
    def get_candidate_info(self, feed_dict, all_item=True):
        '''
        Add entire item pool as candidate for the feed_dict
        @input:
        - all_item: whether obtain all item features from candidate pool
        - feed_dict
        @output:
        - candidate_info: {'item_id': (L,), 
                           'if_{feature_name}': (n_item, feature_dim)}
        '''
        if all_item:
            candidate_info = {'item_id': self.candidate_iids}
            candidate_info.update(self.candidate_item_meta)
        else:
            candidate_info = {'item_id': feed_dict['item_id']}
            indices = feed_dict['item_id'] - 1
            candidate_info.update({k: v[indices] for k,v in self.candidate_item_meta.items()})
        return candidate_info
        
    
        
    def reset(self, params = {'empty_history': True}):
        '''
        Reset environment with new sampled users
        @input:
        - params: {'batch_size': scalar, 
                   'empty_history': True if start from empty history, 
                   'initial_history': start with initial history}
        @process:
        - self.batch_iter
        - self.current_observation
        - self.current_step
        - self.current_temper
        - self.env_history
        @output:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H, feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B, )}}
        '''
        if 'empty_history' not in params:
            params['empty_history'] = False
            
        # set inference batch size
        if 'batch_size' in params:
            BS = params['batch_size']
        else:
            BS = self.episode_batch_size
        
        # random sample users
        self.batch_iter = iter(DataLoader(self.reader, batch_size = BS, shuffle = True, 
                                          pin_memory = True, num_workers = 8))
        sample_info = next(self.batch_iter)
        self.sample_batch = self.get_observation_from_batch(sample_info)
        self.current_observation = self.sample_batch
        self.current_step = torch.zeros(self.episode_batch_size).to(self.device)
        self.current_sample_head_in_batch = BS
        
        # user temper for leave model
        self.current_temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
        self.current_sum_reward = torch.zeros(self.episode_batch_size).to(self.device)

        # batch-wise monitor
        self.env_history = {'step': [0.], 'leave': [], 'temper': [],
                            'coverage': [], 'ILD': []}
        
        return deepcopy(self.current_observation)
        
    
    def step(self, step_dict):
        '''
        users react to the recommendation action
        @input:
        - step_dict: {'action': (B, slate_size),
                      'action_features': (B, slate_size, item_dim) }
        @output:
        - new_observation: {'user_profile': {'user_id': (B,), 
                                             'uf_{feature_name}': (B, feature_dim)}, 
                            'user_history': {'history': (B, max_H), 
                                             'history_if_{feature_name}': (B, max_H, feature_dim), 
                                             'history_{response}': (B, max_H), 
                                             'history_length': (B, )}}
        - response_dict: {'immediate_response': see self.get_response@output - response_dict,
                          'done': (B,)}
        - update_info: see self.update_observation@output - update_info
        '''
        
        # URM forward
        with torch.no_grad():
            action = step_dict['action'] # must be indices on candidate_ids
            
            # get user response
            response_dict = self.get_response(step_dict)
            response = response_dict['immediate_response']

            # done mask and temper update
            # (B,)
            done_mask = self.get_leave_signal(None, None, response_dict) # this will also change self.current_temper
            response_dict['done'] = done_mask
            
            # update user history in current_observation
            # {'slate': (B, slate_size), 'updated_observation': a copy of self.current_observation}
            update_info = self.update_observation(None, action, response, done_mask)
            
            # env_history update: step, leave, temper, converage, ILD
            self.current_step += 1
            n_leave = done_mask.sum()
            self.env_history['leave'].append(n_leave.item())
            self.env_history['temper'].append(torch.mean(self.current_temper).item())
            self.env_history['coverage'].append(response_dict['coverage'])
            self.env_history['ILD'].append(response_dict['ILD'])

            # when users left, new users come into the running batch
            if n_leave > 0:
                final_steps = self.current_step[done_mask].detach().cpu().numpy()
                for fst in final_steps:
                    self.env_history['step'].append(fst)

                if self.current_sample_head_in_batch + n_leave < self.episode_batch_size:
                    # reuse previous batch if there are sufficient samples for n_leave
                    head = self.current_sample_head_in_batch
                    tail = self.current_sample_head_in_batch + n_leave
                    for obs_key in ['user_profile', 'user_history']:
                        for k,v in self.sample_batch[obs_key].items():
                            self.current_observation[obs_key][k][done_mask] = v[head:tail]
                    self.current_sample_head_in_batch += n_leave
                else:
                    # sample new users to fill in the blank
                    sample_info = self.sample_new_batch_from_reader()
                    self.sample_batch = self.get_observation_from_batch(sample_info)
                    for obs_key in ['user_profile', 'user_history']:
                        for k,v in self.sample_batch[obs_key].items():
                            self.current_observation[obs_key][k][done_mask] = v[:n_leave]
                    self.current_sample_head_in_batch = n_leave
                self.current_step[done_mask] *= 0
                self.current_temper[done_mask] *= 0
                self.current_temper[done_mask] += self.initial_temper
            else:
                self.env_history['step'].append(self.env_history['step'][-1])

        return deepcopy(self.current_observation), response_dict, update_info

    def get_response(self, step_dict):
        '''
        @input:
        - step_dict: {'action': (B, slate_size)}
        @output:
        - response_dict: {'immediate_response': (B, slate_size, n_feedback),
                          'user_state': (B, gt_state_dim),
                          'coverage': scalar,
                          'ILD': scalar}
        '''

        # ============================================================
        # [Original code]  这里有bug，做了两次sigmoid
        # ============================================================
        # action = step_dict['action']
        # coverage = len(torch.unique(action))
        # B = action.shape[0]
        
        # batch = {'item_id': self.candidate_iids[action]}
        # batch.update(self.current_observation['user_profile'])
        # batch.update(self.current_observation['user_history'])
        # batch.update({k: v[action] for k, v in self.candidate_item_meta.items()})
        
        # out_dict = self.immediate_response_model(batch)
        
        # behavior_scores = out_dict['probs']                         # 这里命名误导：现在 probs 已经是真概率
        # item_enc = self.candidate_item_encoding[action].view(B, self.slate_size, -1)
        # item_enc_norm = F.normalize(item_enc, p=2.0, dim=-1)
        # corr_factor = self.get_intra_slate_similarity(item_enc_norm)
        
        # point_scores = torch.sigmoid(behavior_scores) - corr_factor.view(B, self.slate_size, 1) * self.rho
        # point_scores[point_scores < 0] = 0
        # response = torch.bernoulli(point_scores).detach()
        
        # return {'immediate_response': response,
        #         'user_state': out_dict['state'],
        #         'coverage': coverage,
        #         'ILD': 1 - torch.mean(corr_factor).item()}
    
        # ============================================================
        # [Fixed + gated code] 修正版：避免 double-sigmoid + click gating
        # ============================================================
    
        # actions (exposures), (B, slate_size), indices of self.candidate_iids
        action = step_dict["action"]
        coverage = len(torch.unique(action))
        B = action.shape[0]
        slate = action.shape[1] if action.dim() == 2 else 1  # 更稳：别完全依赖 self.slate_size
    
        # build URM batch
        batch = {"item_id": self.candidate_iids[action]}
        batch.update(self.current_observation["user_profile"])
        batch.update(self.current_observation["user_history"])
        batch.update({k: v[action] for k, v in self.candidate_item_meta.items()})
    
        out_dict = self.immediate_response_model(batch)
    
        # intra-slate similarity (用于惩罚 & ILD)
        item_enc = self.candidate_item_encoding[action].view(B, slate, -1)
        item_enc_norm = F.normalize(item_enc, p=2.0, dim=-1)
        corr_factor = self.get_intra_slate_similarity(item_enc_norm)  # (B, slate)
    
        # 1) 取概率：优先用 probs（已在[0,1]），否则用 preds(logits) -> sigmoid
        if "probs" in out_dict:
            p = out_dict["probs"]
        elif "preds" in out_dict:
            p = torch.sigmoid(out_dict["preds"])
        else:
            raise KeyError(f"URM output has no preds/probs. keys={list(out_dict.keys())}")
    
        # 2) 同质化惩罚（rho=0 时不影响 p，但 corr_factor 仍用于 ILD）
        if getattr(self, "rho", 0.0) != 0:
            p = p - corr_factor.view(B, slate, 1) * self.rho
    
        # 3) clamp 到 [0,1]
        p = p.clamp(0.0, 1.0)
        # ============================================================
        # [NEG-SAMPLE CALIBRATION] 还原真实行为概率，避免多行为比例虚高
        # ============================================================
        if getattr(self, "enable_neg_sample_calibration", True):
            # build (1,1,n_feedback) keep-rate vector aligned with self.response_types
            if (not hasattr(self, "_neg_keep_vec")) or (self._neg_keep_vec.device != p.device) or (self._neg_keep_vec.dtype != p.dtype):
                # reader里这个就是你统计出来的 pos/neg（is_hate 可能带负号，所以取 abs）
                rates = getattr(self.reader, "response_neg_sample_rate", None)
                if rates is None:
                    rates = self.reader.get_response_weights()
        
                keep = []
                for name in self.response_types:
                    v = float(rates.get(name, 1.0))
                    keep.append(abs(v))  # hate 取绝对值
                self._neg_keep_vec = torch.tensor(keep, device=p.device, dtype=p.dtype).view(1, 1, -1)
        
            s = self._neg_keep_vec  # (1,1,n_feedback)
            # p' -> p_real
            eps = 1e-8
            p = p.clamp(eps, 1.0 - eps)
            s = s.clamp(min=eps)
            p = (s * p) / ((1.0 - p) + s * p)
            p = p.clamp(0.0, 1.0)
    
        # ------------------------------------------------------------
        # 4) click gating：先采样 click，再决定其他行为是否允许发生
        # ------------------------------------------------------------
        # 找到各行为在最后一维的 index（尽量兼容）
        # 你当前的顺序通常是：
        # ['is_click','long_view','is_like','is_comment','is_forward','is_follow','is_hate']
        feedback_types = None
        if hasattr(self.immediate_response_model, "feedback_types"):
            feedback_types = list(self.immediate_response_model.feedback_types)
        elif hasattr(self, "response_types"):
            feedback_types = list(self.response_types)
        else:
            try:
                feedback_types = list(self.reader.get_statistics()["feedback_type"])
            except Exception:
                feedback_types = None
    
        if feedback_types is None:
            # 兜底：假设固定顺序（和你现在 KuaiRand reader 一致）
            feedback_types = ["is_click", "long_view", "is_like", "is_comment", "is_forward", "is_follow", "is_hate"]
    
        idx = {name: i for i, name in enumerate(feedback_types)}
    
        if "is_click" not in idx:
            raise KeyError(f"feedback_types has no is_click. feedback_types={feedback_types}")
    
        click_i = idx["is_click"]
    
        # 初始化 response
        response = torch.zeros_like(p)
    
        # 4.1 先采样 click
        click_p = p[..., click_i]                      # (B, slate)
        click = torch.bernoulli(click_p).detach()      # (B, slate) 0/1
        response[..., click_i] = click
    
        # gating mask：只有 click=1 才允许其他行为发生
        gate = click  # 0/1 float tensor (B, slate)
    
        # 4.2 long_view（如果存在）也 gate
        if "long_view" in idx:
            lv_i = idx["long_view"]
            lv_p = (p[..., lv_i] * gate).clamp(0.0, 1.0)
            response[..., lv_i] = torch.bernoulli(lv_p).detach()
    
        # 4.3 其他行为：like/comment/forward/follow/hate 全部 gate
        for name in ["is_like", "is_comment", "is_forward", "is_follow", "is_hate"]:
            if name in idx:
                i = idx[name]
                pp = (p[..., i] * gate).clamp(0.0, 1.0)
                response[..., i] = torch.bernoulli(pp).detach()
        # --- coverage (step) ---
        flat = action.view(-1)
        step_cov_cnt = torch.unique(flat).numel()
        step_cov_ratio = step_cov_cnt / float(flat.numel())
        
        # --- catalog coverage (running) ---
        if not hasattr(self, "_cov_mask") or self._cov_mask.numel() != self.n_candidate:
            self._cov_mask = torch.zeros(self.n_candidate, dtype=torch.bool, device=action.device)
        self._cov_mask[flat] = True
        catalog_cov_ratio = self._cov_mask.float().mean().item()  # in [0,1]
                
        return {
            "immediate_response": response,
            "user_state": out_dict["state"],
            "coverage": step_cov_ratio,
            "ILD": 1 - torch.mean(corr_factor).item(),
           "coverage_ratio": float(step_cov_ratio), # 新的：每步去重率
           "catalog_coverage": float(catalog_cov_ratio), # 新的：全程目录覆盖率
        }
        

        #这个是修复了两次sigmoid的bug，但没做click多行为处理
        # # actions (exposures), (B, slate_size), indices of self.candidate_iid
        # action = step_dict['action']
        # coverage = len(torch.unique(action))
        # B = action.shape[0]

        # ########################################
        # # This is where the action take effect #
        # ########################################
        # batch = {'item_id': self.candidate_iids[action]}
        # batch.update(self.current_observation['user_profile'])
        # batch.update(self.current_observation['user_history'])
        # batch.update({k: v[action] for k, v in self.candidate_item_meta.items()})

        # out_dict = self.immediate_response_model(batch)

        # # (B, slate_size, item_dim)
        # item_enc = self.candidate_item_encoding[action].view(B, self.slate_size, -1)
        # item_enc_norm = F.normalize(item_enc, p=2.0, dim=-1)
        # # (B, slate_size)
        # corr_factor = self.get_intra_slate_similarity(item_enc_norm)

        # # 1) URM 输出通常同时包含 logits(preds) 与 概率(probs)
        # #    你的 debug 已确认：out_dict['probs'] 已经在 [0,1]，如果再 sigmoid 会导致 double-sigmoid。
        # if "probs" in out_dict:
        #     p = out_dict["probs"]
        # elif "preds" in out_dict:
        #     p = torch.sigmoid(out_dict["preds"])
        # else:
        #     raise KeyError(f"URM output has no preds/probs. keys={list(out_dict.keys())}")

        # # 2) slate 同质化惩罚（rho=0 时不影响 p，但 corr_factor 仍用于 ILD）
        # if self.rho != 0:
        #     p = p - corr_factor.view(B, self.slate_size, 1) * self.rho

        # # 3) 合法化到 [0,1]
        # p = p.clamp(0.0, 1.0)

        # # 4) 采样
        # response = torch.bernoulli(p).detach()

        # return {'immediate_response': response,
        #         'user_state': out_dict['state'],
        #         'coverage': coverage,
        #         'ILD': 1 - torch.mean(corr_factor).item()}

    def get_ground_truth_user_state(self, profile, history):
        batch_data = {}
        batch_data.update(profile)
        batch_data.update(history)
        gt_state_dict = self.immediate_response_model.encode_state(batch_data, self.episode_batch_size)
        gt_user_state = gt_state_dict['state'].view(self.episode_batch_size,1,self.gt_state_dim)
        return gt_user_state
    
    def get_intra_slate_similarity(self, action_item_encoding):
        '''
        @input:
        - action_item_encoding: (B, slate_size, enc_dim)
        @output:
        - similarity: (B, slate_size)
        '''
        B, L, d = action_item_encoding.shape
        # pairwise similarity in a slate (B, L, L)
        pair_similarity = torch.mean(action_item_encoding.view(B,L,1,d) * action_item_encoding.view(B,1,L,d), dim = -1)
        # similarity to slate average, (B, L)
        point_similarity = torch.mean(pair_similarity, dim = -1)
        return point_similarity
    
    def get_leave_signal(self, user_state, action, response_dict):
        '''
        User leave model maintains the user temper, and a user leaves when the temper drops below 1.
        @input:
        - user_state: not used in this env
        - action: not used in this env
        - response_dict: (B, slate_size, n_feedback)
        @process:
        - update temper
        @output:
        - done_mask: 
        '''
        # (B, slate_size, n_feedback)
        point_reward = response_dict['immediate_response'] * self.response_weights.view(1,1,-1)
        # (B, slate_size)
        combined_reward = torch.sum(point_reward, dim = 2)
        # (B, )
        temper_boost = torch.mean(combined_reward, dim = 1)
        # temper update for leave model
        temper_update = temper_boost - 2
        temper_update[temper_update > 0] = 0
        temper_update[temper_update < -2] = -2
        self.current_temper += temper_update
        # leave signal
        done_mask = self.current_temper < 1
        return done_mask
        
    def update_observation(self, user_state, action, response, done_mask, update_current = True):
        '''
        user profile stays static, only update user history
        @input:
        - user_state: not used in this env
        - action: (B, slate_size), indices of self.candidate_iids
        - response: (B, slate_size, n_feedback)
        - done_mask: not used in this env
        @output:
        - update_info: {slate: (B, slate_size),
                        updated_observation: same format as self.reset@output - observation}
        '''
        # (B, slate_size), convert to encoded item id
        rec_list = self.candidate_iids[action]
        
        # history update
        old_history = self.current_observation['user_history']
        max_H = self.max_hist_len
        L = old_history['history_length'] + self.slate_size
        L[L>max_H] = max_H
        new_history = {'history': torch.cat((old_history['history'], rec_list), dim = 1)[:,-max_H:], 
                       'history_length': L}
        # history item features
        for k,candidate_meta_features in self.candidate_item_meta.items():
            # (B, slate_size, feature_dim)
            meta_features = candidate_meta_features[action]
            # (B, max_H, feature_dim)
            previous_meta = old_history[f'history_{k}'].view(self.episode_batch_size, max_H, -1)
            new_history[f'history_{k}'] = torch.cat((previous_meta, meta_features), dim = 1)[:,-max_H:,:].view(self.episode_batch_size,-1)
        # history item responses
        for i,R in enumerate(self.immediate_response_model.feedback_types):
            k = f'history_{R}'
            new_history[k] = torch.cat((old_history[k], response[:,:,i]), dim = 1)[:,-max_H:]
        if update_current:
            self.current_observation['user_history'] = new_history
        return {'slate': rec_list, 'updated_observation': {
                                        'user_profile': deepcopy(self.current_observation['user_profile']), 
                                        'user_history': deepcopy(new_history)}}
    
    def sample_new_batch_from_reader(self):
        '''
        @output
        - sample_info: see BaseRLEnvironment.get_observation_from_batch@input - sample_batch
        '''
        new_sample_flag = False
        try:
            sample_info = next(self.batch_iter)
            if sample_info['user_profile'].shape[0] != self.episode_batch_size:
                new_sample_flag = True
        except:
            new_sample_flag = True
        if new_sample_flag:
            self.batch_iter = iter(DataLoader(self.reader, batch_size = self.episode_batch_size, shuffle = True, 
                                              pin_memory = True, num_workers = 8))
            sample_info = next(self.batch_iter)
        return sample_info
    
    def stop(self):
        self.batch_iter = None
    
    def get_new_iterator(self, B):
        return iter(DataLoader(self.reader, batch_size = B, shuffle = True, 
                               pin_memory = True, num_workers = 8))
    

    def create_observation_buffer(self, buffer_size):
        '''
        @input:
        - buffer_size: L, scalar
        @output:
        - observation: {'user_profile': {'user_id': (L,), 
                                         'uf_{feature_name}': (L, feature_dim)}, 
                        'user_history': {'history': (L, max_H), 
                                         'history_if_{feature_name}': (L, max_H * feature_dim), 
                                         'history_{response}': (L, max_H), 
                                         'history_length': (L,)}}
        '''
        observation = {'user_profile': {'user_id': torch.zeros(buffer_size).to(torch.long).to(self.device)}, 
                       'user_history': {'history': torch.zeros(buffer_size, self.max_hist_len).to(torch.long).to(self.device), 
                                        'history_length': torch.zeros(buffer_size).to(torch.long).to(self.device)}}
        for f,f_dim in self.observation_space['user_feature_dims'].items():
            observation['user_profile'][f'uf_{f}'] = torch.zeros(buffer_size, f_dim).to(torch.float).to(self.device)
        for f,f_dim in self.observation_space['item_feature_dims'].items():
            observation['user_history'][f'history_if_{f}'] = torch.zeros(buffer_size, f_dim * self.max_hist_len)\
                                                                                .to(torch.float).to(self.device)
        for f in self.observation_space['feedback_type']:
            observation['user_history'][f'history_{f}'] = torch.zeros(buffer_size, self.max_hist_len)\
                                                                                .to(torch.float).to(self.device)
        return observation
    
    def get_report(self, smoothness = 10):
        return {k: np.mean(v[-smoothness:]) for k,v in self.env_history.items()}