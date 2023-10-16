#!/bin/bash
WORK_DIR=$(cd `dirname $0`; pwd)/..

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

cd $WORK_DIR


torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_addr localhost \
--master_port 34215 \
demo.py --amp $@
