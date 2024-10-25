import yaml
import os
import pickle
from rewards.system import RewardingSystem
from algorithm.base import Item
import numpy as np
import pygmo as pg
from tdc.generation import MolGen
data = MolGen(name='ZINC')
split = data.get_data()
zinc_smiles = split.smiles.values

system = RewardingSystem(use_tqdm=True,chunk_size=1000)
property_list = ['sa','drd2','qed','gsk3b','jnk3']
def cal_hv(smiles,property_list):
    inzinc = 0
    for i in smiles:
        if i in zinc_smiles:
            inzinc += 1
    combined_smiles = [[i,i] for i in smiles]
    results = system.evaluate(property_list,combined_smiles)
    for p in results.keys():
        if p in ['qed','jnk']: # max
            results[p] = 1-np.array(results[p])
        if p == 'sa': # min
            results[p] = (np.array(results[p]) - 1)/9
    results = [results[p] for p in property_list]
    results = np.stack(results).transpose()
    print(results.shape)
    hv_pygmo = pg.hypervolume(results)
    hypervolume = hv_pygmo.compute(np.array([1.0 for i in range(len(property_list))]))
    print('hv is:',hypervolume,f'total {len(smiles)} mols, {inzinc} moles is in zinc')

def cal_hv_two():
    file_path = "/home/v-nianran/src/MOLLEO/multi_objective/main/molleo_multi_pareto/results/results_GPT-4_['jnk3', 'qed']_['sa', 'gsk3b', 'drd2']1.yaml"
    
    with open(file_path, 'r') as f:
        mol_buffer = yaml.load(f, Loader=yaml.FullLoader)
    top100_leo = sorted(mol_buffer.items(), key=lambda item: item[1][0], reverse=True)[:]
    top100_leo = [i[0] for i in top100_leo]
    cal_hv(top100_leo,property_list)
    cal_best_zinc(top100_leo,property_list)

    file_path = "/home/v-nianran/src/results/sa_drd2_qed_gsk3b_jnk3_goals5_44.pkl"
    with open(file_path,'rb') as f:
        obj = pickle.load(f)
    # dict_keys(['history', 'init_pops', 'final_pops', 'all_mols', 'properties', 'evaluation', 'running_time'])
    top100 = sorted(obj['all_mols'], key=lambda item: item.total, reverse=True)[:len(top100_leo)]
    top100 = [i.value for i in top100]
    cal_hv(top100,property_list)
    cal_best_zinc(top100,property_list)

def cal_best_zinc(smiles,property_list):
    combined_smiles = [[i,i] for i in smiles]
    results = system.evaluate(property_list,combined_smiles)
    for p in results.keys():
        if p in ['gsk3b','drd2']: # max
            results[p] = 1-np.array(results[p])
        if p == 'sa': # min
            results[p] =1 - (np.array(results[p]) - 1)/9
    results = [results[p] for p in property_list]
    results = np.stack(results).transpose()
    results = results.sum(axis = 1)
    results = np.sort(results)[::-1]
    print('top 10',results[:10]) # [4.32860274 4.17765446 4.1741578  4.14235197 4.1084562  4.10729573, 4.08959373 4.06922041 4.06246599 4.06169614]

cal_hv_two()

cal_best_zinc(zinc_smiles,property_list)

    
    
    

