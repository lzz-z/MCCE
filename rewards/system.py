import requests
import json
url = 'http://cpu1.ms.wyue.site:8000/process'
import time

import numpy as np
from tqdm import tqdm
def get_evaluation(evaluate_metric, smiles):
    data = {
        "ops": evaluate_metric,
        "data":smiles
    }
    response = requests.post(url, json=data)
    result = response.json()['results']
    return result

from tdc import Oracle, Evaluator
class RewardingSystem:
    def __init__(self,use_tqdm=False,chunk_size=20,material=False):
        tdc_func = ['GSK3B','JNK3','DRD2','SA',
                    'QED','LogP','Celecoxib_Rediscovery','Troglitazone_Rediscovery',
                    'Thiothixene_Rediscovery',
                    'Aripiprazole_Similarity','Median 1','Median 2',
                    'Isomers_C7H8N2O2','Isomers_C9H10N2O2PF2CL',
                    'Osimertinib_MPO','Fexofenadine_MPO','Ranolazine_MPO',
                    'Perindopril_MPO','Amlodipine_MPO','Sitagliptin_MPO',
                    'Zaleplon_MPO','Valsartan_SMARTS','Scaffold Hop',]
        tdc_evaluator = ['Diversity','Uniqueness','Validity','Novelty']
        self.all_rewards = {
            name.lower():Oracle(name=name) for name in tdc_func
        }
        self.all_rewards['similarity'] = morgan_similarity
        self.all_rewards['donor'] = donor_number
        if material:
            from yue.objectives import AnolyteLabeler
            self.remote_labeler = AnolyteLabeler()
            self.all_rewards['logs'] = self.remote_labeler.log_s_endpoint.observe
            self.all_rewards['smarts_filter'] = self.remote_labeler.get_filter_results
            self.all_rewards['reduction_potential'] = self.remote_labeler.red_pot_endpoint.observe
        self.all_evaluators={
            name.lower():Evaluator(name=name) for name in tdc_evaluator
        }
        self.use_tqdm = use_tqdm
        self.chunk_size=chunk_size
    

    def register_reward(self, reward_name, reward_function):
        self.all_rewards[reward_name] = reward_function

    def get_reward(self, reward_name, items):
        if reward_name in ['similarity']:
            new_items = items
        else:
            new_items = [i[1] for i in items]

        results = []
        chunk_size = self.chunk_size
        if self.use_tqdm:
            for i in tqdm(range(0, len(new_items), chunk_size)):
                chunk = new_items[i:i+chunk_size]
                result = self.all_rewards[reward_name](chunk)
                results.append(result)
        else:
            for i in range(0, len(new_items), chunk_size):
                chunk = new_items[i:i+chunk_size]
                result = self.all_rewards[reward_name](chunk)
                results.append(result)

        return np.concatenate(results)
    
    def evaluate(self,ops,smiles_list):
        results = {}
        for op in ops:
            r = self.get_reward(op,smiles_list)
            results[op] = r
        return results
    '''
    def evaluate(self,ops,smiles_list):
        while True:
            try:
                return get_evaluation(ops,smiles_list)
            except Exception as e:
                print(f'encounter exception in get evaluation: {e}, retry in 60s')
                time.sleep(60)
    '''
    def evaluate_items(self, items, qed_requ, logp_requ, donor_requ, donor_num):
        smiles_list = [[item.value] for item in items]
        fitnesses, donors, logps, qeds = self._get_evaluation(smiles_list, qed_requ, logp_requ, donor_requ, donor_num)

        for i, item in enumerate(items):
            item.scores = {'qed': qeds[i], 'logp': logps[i], 'donor': donors[i]}
            item.fitness = fitnesses[i]

    def _get_evaluation(self, smiles_list, qed_requ, logp_requ, donor_requ, donor_num):
        # 具体的评估函数与之前的类似
        pass
    
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def morgan_similarity(items):
    results = [_morgan_similarity(i[0],i[1]) for i in items]
    return results

def _morgan_similarity(a, b):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2) 

from rdkit import Chem
from rdkit.Chem import Lipinski

def calculate_donor_number(smiles: str) -> int:
    """
    :return: The number of hydrogen bond donors
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    # Use Lipinski's definition to calculate the number of H-bond donors
    donor_number = Lipinski.NumHDonors(mol)
    return donor_number
def donor_number(smiles_list):
    return [calculate_donor_number(i) for i in smiles_list]

if __name__ == '__main__':
    s = RewardingSystem()
    mols = [['CCH','CCOCCOCC'],['CCOCCOCC','C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']]
    ops = ['qed','logp','similarity','donor','sa','reduction_potential','smarts_filter','logs']
    
    print(s.evaluate_new(ops,mols))