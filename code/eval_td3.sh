#!/bin/bash
SEED=2026
ENV_CLASS='KRCrossSessionEnvironment_GPU'
POLICY_CLASS='ActionTransformer'

ACTOR_PATH="output/Kuairand_Pure/agents/td3_ActionTransformer_KRCrossSessionEnvironment_GPU_actor0.0001_critic0.001_niter30000_reg0.00001_ep0.01_noise0.1_bs128_epbs32_fret0.1_seed${SEED}/model_actor"

python eval_td3.py \
  --cuda 0 \
  --seed ${SEED} \
  --env_class ${ENV_CLASS} \
  --policy_class ${POLICY_CLASS} \
  --actor_path ${ACTOR_PATH} \
  --n_eval_steps 1000 \
  --retention_threshold 3.0 \
  --uirm_log_path output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.log \
  --max_n_session 6 \
  --max_return_day 10 \
  --initial_temper 20 \
  --next_day_return_bias 0.4 \
  --feedback_influence_on_return 0.1 \
  --slate_size 6 \
  --episode_batch_size 32 \
  --item_correlation 0.2 \
  --max_step_per_episode 20 \
  --policy_user_latent_dim 16 \
  --policy_item_latent_dim 16 \
  --policy_enc_dim 32 \
  --policy_attn_n_head 4 \
  --policy_transformer_d_forward 64 \
  --policy_transformer_n_layer 2 \
  --policy_hidden_dims 128 \
  --policy_dropout_rate 0.1
