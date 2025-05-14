#!/bin/bash

# Define the Python scripts to run in the order
commands=(
  #"python main.py yue/goal5_best.yaml --seed 44"
  #"python main.py yue/goal5_best.yaml --seed 45"
  #"python main.py yue/goal5_best.yaml --seed 46"
  #"python main.py yue/goal5.yaml --seed 44"
  "python main.py yue/goal5.yaml --seed 45"
  "python main.py yue/goal5.yaml --seed 46"
  
)

# Print and execute each command
for cmd in "${commands[@]}"; do
  echo "Executing: $cmd"   # Print the command
  $cmd                    # Execute the command
done
