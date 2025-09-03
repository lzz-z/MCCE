#!/bin/bash

# Define the list of objectives
objectives=(
  #'qed'
  'jnk3' 
  #'drd2'
  'gsk3b'
  'mestranol_similarity'
  'albuterol_similarity'
  'thiothixene_rediscovery'
  'celecoxib_rediscovery'
  'troglitazone_rediscovery'
  'perindopril_mpo'
  'ranolazine_mpo'
  'sitagliptin_mpo'
  'amlodipine_mpo'
  'fexofenadine_mpo'
  'osimertinib_mpo'
  'zaleplon_mpo'
  'median1'
  'median2'
  'isomers_c7h8n2o2'
  'isomers_c9h10n2o2pf2cl'
  'deco_hop'
  'scaffold_hop'
  'valsartan_smarts'
)

# Loop over each objective and each seed
for objective in "${objectives[@]}"; do
  # Determine the starting seed based on the objective
  start_seed=42
  if [ "$objective" == "" ]; then
    start_seed=43 #  already run for seed 42, so start from 43
  fi

  for seed in $(seq "$start_seed" 46); do
    echo "Executing: python main.py molecules/config.yaml --seed $seed --objective $objective --directions max"
    python main.py molecules/config.yaml --seed "$seed" --objective "$objective" --directions max
  done
done