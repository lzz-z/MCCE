#!/bin/bash

# Define the list of objectives
objectives=(
  'qed' 
  #'drd2'
  'jnk3'
  'gsk3b'
  #'isomers_c9h10n2o2pf2cl'
  #'osimertinib_mpo'
  #'ranolazine_mpo'
  #'perindopril_mpo'
  #'celecoxib_rediscovery'
  #'thiothixene_rediscovery'
  #'albuterol_similarity'
  #'mestranol_similarity'
  #'median1'
  #'median2'
  #'isomers_c7h8n2o2'
  #'fexofenadine_mpo'
  #'amlodipine_mpo'
  #'sitagliptin_mpo'
  #'zaleplon_mpo'
  #'troglitazone_rediscovery'
  #'deco_hop'
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