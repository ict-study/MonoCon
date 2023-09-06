#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

# mkdir /mnt/vepfs/Perception/perception-users
# mkdir /mnt/vepfs/Perception/perception-public
# if ! [ -d "/vepfs" ]; then
#   ln -s /qcraft-vepfs-01 /vepfs
# fi
# if ! [ -d "/mnt/vepfs/Perception/perception-users" ]; then
#   ln -s /qcraft-vepfs-01/Perception/perception-users /mnt/vepfs/Perception/perception-users
# fi
# if ! [ -d "/mnt/vepfs/Perception/perception-public" ]; then
#   ln -s /qcraft-vepfs-01/Perception/perception-public /mnt/vepfs/Perception/perception-public
# fi
# if ! [ -d "/tos/qcraftlabeldata" ]; then
#   ln -s /tos/qcraft/qcraftlabeldata /tos/qcraftlabeldata
# fi

#find . -name '*.so' -delete

source /mnt/vepfs/Perception/perception-users/hongliang/condanew/anaconda3/bin/activate \
  /mnt/vepfs/Perception/perception-users/hongliang/condanew/anaconda3/envs/monoconv



#python -c "import mmdet3d; print('mmdet3d path:'); mmdet3d"

# export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1; NCCL_DEBUG=INFO

# PYTHONPATH="/root/code/MonoCon/mmdetection3d-0.14.0":"/root/code/MonoCon/mmdetection-2.11.0":$PYTHONPATH \
# python -c "import mmdet3d; mmdet3d.__path__"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
PYTHONPATH="/root/code/MonoCon/mmdetection3d-0.14.0":"/root/code/MonoCon/mmdetection-2.11.0":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

# PYTHONPATH="/root/code/MonoCon/mmdetection3d-0.14.0":"/root/code/MonoCon/mmdetection-2.11.0":$PYTHONPATH \
# OMP_NUM_THREADS=4 python3 -m torch.distributed.launch --nproc_per_node "$MLP_WORKER_GPU" \
#   --master_addr "$MLP_WORKER_0_HOST" \
#   --master_port "$MLP_WORKER_0_PORT" \
#   --node_rank "$MLP_ROLE_INDEX" \
#   --nnodes="$MLP_WORKER_NUM" /root/code/MonoCon/mmdetection3d-0.14.0/tools/train.py \
#   --launcher=pytorch "${@:1}"