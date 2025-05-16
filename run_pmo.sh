#!/bin/bash

# Define the list of objectives
objectives=(

  # 'drd2' 'jnk3' 'gsk3b' 'isomers_c9h10n2o2pf2cl' 'osimertinib_mpo'
  # 'ranolazine_mpo' 'perindopril_mpo'
  #'qed' 'celecoxib_rediscovery' 'thiothixene_rediscovery' 
  'albuterol_similarity' 'mestranol_similarity' 'median1' 'median2' 
  'isomers_c7h8n2o2' 'fexofenadine_mpo' 'amlodipine_mpo'
  'sitagliptin_mpo' 'zaleplon_mpo' 'valsartan_smarts' 'deco_hop' 'scaffold_hop' 
  # 'troglitazone_rediscovery'
)

# need more exp: gsk3b, valsartan_smarts random2

# Loop over each objective and run the experiment
for objective in "${objectives[@]}"; do
  echo "Executing: python main.py yue/goal5.yaml --seed 42 --objective $objective"
  python main.py yue/goal5.yaml --seed 42 --objective "$objective"
done
