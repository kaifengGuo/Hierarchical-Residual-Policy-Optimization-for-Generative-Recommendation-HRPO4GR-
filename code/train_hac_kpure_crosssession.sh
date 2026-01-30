#!/bin/bash
set -e

# ========== 通用路径 ==========
ROOT_PATH="/root/KuaiSim-main"
cd ${ROOT_PATH}/code

mkdir -p output
mkdir -p output/Kuairand_Pure/
mkdir -p output/Kuairand_Pure/agents/

env_path="output/Kuairand_Pure/env/"
output_path="output/Kuairand_Pure/agents"

# 你之前 user response 预训练环境的 log 名字
log_name="user_KRMBUserResponse_lr0.0001_reg0_nlayer2"

# ========== 环境参数 ==========
ENV_CLASS='KRCrossSessionEnvironment_GPU'
MAX_SESSION=6               # 每个 episode 的最大 session 数
MAX_RET_DAY=10              # 最大回访天数（截断）
RET_DAY_BIAS=0.4            # 几何分布的 base，让 “第二天回来” 更常见
FEEDBACK_INF_RETURN=0.1     # 当前反馈对后续回访的影响系数

SLATE_SIZE=6
EP_BS=32
RHO=0.2                     # item 相关性
MAX_STEP=20
INITIAL_TEMPER=20

# ========== HAC / DDPG 超参数 ==========
ENV_SEED=11
CUDA_ID=0

GAMMA=0.8
REG=0.00001

ACTOR_LR=0.0001
CRITIC_LR=0.001

N_ITER=30000
TRAIN_EVERY=1
START_TRAIN_AT=0

INITEP=0.01
ELBOW=0.3
EXPLORE_RATE=1.0

BS=128
NOISE=0.1

# ========== 类名（关键） ==========
POLICY_CLASS='OneStageHyperPolicy_with_DotScore'
CRITIC_CLASS='QCritic'
BUFFER_CLASS='ReplayBuffer'
# 如果你已经写了专门的 CrossSessionHAC agent，把这里改成对应类名
AGENT_CLASS='CrossSessionHAC'

file_key="hac_${POLICY_CLASS}_${ENV_CLASS}_actor${ACTOR_LR}_critic${CRITIC_LR}_niter${N_ITER}_reg${REG}_ep${GAMMA}_noise${NOISE}_bs${BS}_epbs${EP_BS}_fret${FEEDBACK_INF_RETURN}_seed${ENV_SEED}"
save_dir="${output_path}/${file_key}/model"
mkdir -p "${save_dir}"

python train_td3.py \
  --seed ${ENV_SEED} \
  --cuda ${CUDA_ID} \
  --env_class ${ENV_CLASS} \
  --policy_class ${POLICY_CLASS} \
  --critic_class ${CRITIC_CLASS} \
  --agent_class ${AGENT_CLASS} \
  --buffer_class ${BUFFER_CLASS} \
  \
  --gamma ${GAMMA} \
  --reward_func get_retention_reward \
  --n_iter ${N_ITER} \
  --train_every_n_step ${TRAIN_EVERY} \
  --start_policy_train_at_step ${START_TRAIN_AT} \
  --initial_epsilon ${INITEP} \
  --final_epsilon ${INITEP} \
  --elbow_epsilon ${ELBOW} \
  --explore_rate ${EXPLORE_RATE} \
  --check_episode 10 \
  --save_episode 200 \
  --save_path ${save_dir} \
  --batch_size ${BS} \
  --noise_var ${NOISE} \
  --noise_clip 1.0 \
  \
  --uirm_log_path ${env_path}log/${log_name}.model.log \
  --max_n_session ${MAX_SESSION} \
  --max_return_day ${MAX_RET_DAY} \
  --next_day_return_bias ${RET_DAY_BIAS} \
  --feedback_influence_on_return ${FEEDBACK_INF_RETURN} \
  --slate_size ${SLATE_SIZE} \
  --episode_batch_size ${EP_BS} \
  --item_correlation ${RHO} \
  --max_step_per_episode ${MAX_STEP} \
  --initial_temper ${INITIAL_TEMPER} \
  \
  --policy_user_latent_dim 16 \
  --policy_item_latent_dim 16 \
  --policy_enc_dim 32 \
  --policy_attn_n_head 4 \
  --policy_transformer_d_forward 64 \
  --policy_transformer_n_layer 2 \
  --policy_hidden_dims 128 \
  --policy_dropout_rate 0.1 \
  > ${output_path}/${file_key}/log
