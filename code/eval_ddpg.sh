#!/bin/bash
mkdir -p output
mkdir -p output/Kuairand_Pure/
mkdir -p output/Kuairand_Pure/agents/

env_path="output/Kuairand_Pure/env/"
output_path="output/Kuairand_Pure/agents"

log_name="user_KRMBUserResponse_lr0.0001_reg0_nlayer2"

########## 环境参数 ##########
ENV_CLASS='KRCrossSessionEnvironment_GPU'
MAX_SESSION=6
MAX_RET_DAY=10
MAX_STEP=20
RET_DAY_BIAS=0.4
FEEDBACK_INF_RETURN=0.1
SLATE_SIZE=6
EP_BS=32
RHO=0.2

########## policy 参数 ##########
POLICY_CLASS='ActionTransformer'

########## 评估用的一些固定超参（要和训练时保持一致） ##########
N_ITER=30000
ACTOR_LR=0.0001
CRITIC_LR=0.001
REG=0.00001
INITEP=0.01
NOISE=0.1
BS=128
SEED=2026

########## 算 retention 用的阈值（<=3 天算留存） ##########
RET_THRESHOLD=3.0

# 根据 train_ddpg_kpure_crosssession.sh 里的命名规则拼接 file_key
file_key=ddpg_${POLICY_CLASS}_${ENV_CLASS}_actor${ACTOR_LR}_critic${CRITIC_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_fret${FEEDBACK_INF_RETURN}_seed${SEED}

ACTOR_PATH="${output_path}/${file_key}/model_actor"

echo "Using actor: ${ACTOR_PATH}"

python eval_td3.py \
  --seed ${SEED} \
  --cuda 0 \
  --env_class ${ENV_CLASS} \
  --policy_class ${POLICY_CLASS} \
  --actor_path ${ACTOR_PATH} \
  --n_eval_steps 1000 \
  --retention_threshold ${RET_THRESHOLD} \
  --uirm_log_path ${env_path}log/${log_name}.model.log \
  --max_n_session ${MAX_SESSION} \
  --max_return_day ${MAX_RET_DAY} \
  --initial_temper ${MAX_STEP} \
  --next_day_return_bias ${RET_DAY_BIAS} \
  --feedback_influence_on_return ${FEEDBACK_INF_RETURN} \
  --slate_size ${SLATE_SIZE} \
  --episode_batch_size ${EP_BS} \
  --item_correlation ${RHO} \
  --max_step_per_episode ${MAX_STEP} \
  --policy_user_latent_dim 16 \
  --policy_item_latent_dim 16 \
  --policy_enc_dim 32 \
  --policy_attn_n_head 4 \
  --policy_transformer_d_forward 64 \
  --policy_transformer_n_layer 2 \
  --policy_hidden_dims 128 \
  --policy_dropout_rate 0.1
