#!/bin/bash
MODE=$1
INIT_WEIGHTS=$2
NAME=exp2

DIR=exp/ptb
MASK=$DIR/mtl/mask
DATA=data/ptb.pkl


mkdir -p $DIR/mtl $DIR/single


if [ "$MODE" = "single" ]; then
  args="  --data_path $DATA
          --seed 2019
          --arch cnn-lstm
          --pruning_iter 20
          --final_rate 0.1
          --epochs 100
          --optim \"sgd(lr=0.1, momentum=0.9)\"
          --hidden_size 200
          --n_layer 2
          --dropout 0.5
          --batch_size 10
          --save_dir $DIR/single
          "

  # --init_weights
  if [ -n "$INIT_WEIGHTS" ]; then
      args=$args" --init_weights $INIT_WEIGHTS"
  fi

  cmd1="CUDA_VISIBLE_DEVICES=0 python train_single.py
      $args
      --task_id 0
      --exp_name 0
      > /dev/null &
      "

  cmd2="CUDA_VISIBLE_DEVICES=0 python train_single.py
      $args
      --task_id 1
      --exp_name 1
      > /dev/null &
      "

  cmd3="CUDA_VISIBLE_DEVICES=0 python train_single.py
      $args
      --task_id 2
      --exp_name 2
      > /dev/null &
      "

  echo $cmd1 && eval $cmd1
  echo $cmd2 && eval $cmd2
  echo $cmd3 && eval $cmd3

elif [ "$MODE" = "mtl" ]; then
  # --init_masks
  if [ -n "$INIT_WEIGHTS" ]; then
      MASK=$INIT_WEIGHTS
  fi

  cmd="python -u train_mtl.py
      --save_dir $DIR/mtl
      --data_path $DATA
      --exp_name $NAME
      --arch cnn-lstm
      --trainer re-seq-label
      --seed 2019
      --print_every 500
      --epochs 100
      --batch_size 10
      --hidden_size 200
      --n_layer 2
      --dropout 0.5
      --optim \"sgd(lr=0.1,momentum=0.9)\"
      --scheduler \"fix\"
      --tasks 0,1,2
      --masks_path $MASK
  "
  echo $cmd && eval $cmd

else
  echo "please choose single or mtl, not: $MODE"
fi