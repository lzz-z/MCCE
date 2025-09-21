#!/bin/bash

# Seeds
seeds=(42 43 44)

# Loop over num_offspring from 6 down to 1
for ((n=6; n>=1; n--)); do
  for seed in "${seeds[@]}"; do
    cmd="python main.py molecules/config_single_gpt.yaml --seed $seed --num_offspring $n --save_suffix ${n}off"
    echo "Executing: $cmd"
    $cmd
  done
done
