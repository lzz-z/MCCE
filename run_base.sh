#!/bin/bash

# Define the Python scripts to run in the order
commands=(
  "python main.py molecules/goal5_gemini.yaml --seed 42"
  "python main.py molecules/goal5_gemini.yaml --seed 43"
  "python main.py molecules/goal5_gemini.yaml --seed 44"
  "python main.py molecules/goal5_gemini.yaml --seed 45"
  "python main.py molecules/goal5_gemini.yaml --seed 46"
  
)

# Print and execute each command
for cmd in "${commands[@]}"; do
  echo "Executing: $cmd"   # Print the command
  $cmd                    # Execute the command
done
