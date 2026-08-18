#!/usr/bin/env bash
# Start volleyball iter31 from locomotion pretrain checkpoint.
# Run this after locomotion pretrain completes.
set -e
cd /home/beast/volleyball-engine

PRETRAIN_CKPT="checkpoints/locomotion/pretrain_v2.pt"
LOG="logs/volleyball_iter31.log"
CKPT_DIR="checkpoints/iter31"

if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "ERROR: $PRETRAIN_CKPT not found — wait for pretrain to finish"
    exit 1
fi

mkdir -p "$CKPT_DIR" recordings

nohup .venv/bin/python -u -m src.training.train \
    --episodes 500000 \
    --n-envs 1 \
    --resume "$PRETRAIN_CKPT" \
    --checkpoint-dir "$CKPT_DIR" \
    --recordings-dir recordings \
    --log-dir runs/iter31 \
    --save-every 1000 \
    --record-every 500 \
    --frozen-update-interval 200 \
    --entropy-coeff 0.02 \
    > "$LOG" 2>&1 &

echo "iter31 PID: $!"
echo $! > training.pid
echo "Tail log: tail -f $LOG"
