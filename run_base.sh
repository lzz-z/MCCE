#!/bin/bash

# Seeds
seeds=(42 43 44)

# Config files
configs=(
  #"molecules/config_exp7.yaml"
  #"molecules/config_exp9.yaml"
  "molecules/config.yaml"
)

# Loop over configs and seeds
for cfg in "${configs[@]}"; do
  for seed in "${seeds[@]}"; do
    cmd="python main.py $cfg --seed $seed"
    echo "Executing: $cmd"
    $cmd
  done
done