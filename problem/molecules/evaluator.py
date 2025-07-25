import requests
import json
url = 'http://cpu1.ms.wyue.site:8000/process'
import time
from functools import partial
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
def get_evaluation(evaluate_metric, smiles):
    data = {
        "objs": evaluate_metric,
        "data":smiles
    }
    response = requests.post(url, json=data)
    result = response.json()['results']
    return result

from tdc import Oracle, Evaluator

def generate_initial_population():
    with open('/home/hp/src/MOLLM/data/data_goal5.json','r') as f:
        data = json.load(f)
    data_type = 'random1' # initial_pop: "random1" # best100, worst100, random1 ~ 5
    print(f'loading {data_type} as initial pop!')
    smiles = data[data_type]
    return smiles

class RewardingSystem:
    def __init__(self,use_tqdm=False,chunk_size=20,config=None):
        tdc_func = ['GSK3B','JNK3','DRD2','SA',
                    'QED','LogP','Celecoxib_Rediscovery','Troglitazone_Rediscovery',
                    'Thiothixene_Rediscovery','albuterol_similarity','mestranol_similarity',
                    'Aripiprazole_Similarity','Median1','Median2',
                    'Isomers_C7H8N2O2','Isomers_C9H10N2O2PF2CL',
                    'Osimertinib_MPO','Fexofenadine_MPO','Ranolazine_MPO',
                    'Perindobjril_MPO','Amlodipine_MPO','Sitagliptin_MPO',
                    'Zaleplon_MPO','Valsartan_SMARTS','scaffold_hobj','deco_hobj']
        
        self.config = config
        self.objs = config.get('goals')
        self.obj_directions = {obj: config.get('optimization_direction')[i] for i, obj in enumerate(self.objs)}

        

        tdc_evaluator = ['Diversity','Uniqueness','Validity','Novelty']
        self.all_rewards = {
            name.lower():Oracle(name=name) for name in tdc_func
        }
        self.all_rewards['similarity'] = morgan_similarity
        self.all_rewards['donor'] = donor_number
        self.all_evaluators={
            name.lower():Evaluator(name=name) for name in tdc_evaluator
        }
        self.use_tqdm = use_tqdm
        self.chunk_size=chunk_size

    def get_reward(self, reward_name, items):
        results = []
        chunk_size = self.chunk_size
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i+chunk_size]
            result = self.all_rewards[reward_name](chunk)
            results.append(result)
        return np.concatenate(results)
    
    def evaluate(self,items):
        items, failed_num,repeated_num = sanitize(items)
        original_results = {}
        transformed_results = {}
        log_dict = {}
        # You should assign the evaluation scores for each item by each item, but you can evaluate all in once
        for obj in self.objs:
            original_results[obj] = self.get_reward(obj,[item.value for item in items])
            transformed_results[obj] = self.transform_objectives(obj,original_results[obj])
        for idx,item in enumerate(items):
            results = {'original_results':{},'transformed_results':{}}
            overall_score = len(self.objs) * 1. # best score
            for obj in self.objs:
                results['original_results'][obj] = original_results[obj][idx]
                results['transformed_results'][obj] = transformed_results[obj][idx]
                overall_score -= results['transformed_results'][obj]
            results['overall_score'] = overall_score # this score cannot be ignore, it is the overall fitness
            item.assign_results(results)
        log_dict['repeated_num'] = repeated_num
        log_dict['invalid_num'] = failed_num
        return items,log_dict

    
    def transform_objectives(self,obj, values):
        values = self.normalize_objectives(obj,values)
        values = self.adjust_direction(obj,values)
        return values 
    
    def normalize_objectives(self,obj,values):
        if obj == 'sa':
            values = (values - 1) / 9
        if obj == 'logs':
            values = (values + 8) / 9
        return values
            
    def adjust_direction(self,obj,values):
        if self.obj_directions[obj] == 'max': 
            # transform to minimization to fit the MOO libraries
            return 1 - values
        elif self.obj_directions[obj] == 'min': 
            return values
        else:
            raise NotImplementedError(f'{obj} is not defined for optimizaion direction! Please define it in "optimization_direction" in your yaml config')

def sanitize(tmp_offspring):
    offspring = []
    failed_num = 0
    repeated_num = 0
    smiles_this_gen = []
    for child in tmp_offspring:
        mol = Chem.MolFromSmiles(child.value) 
        if mol is None: # check if valid
            failed_num += 1
        else:
            child.value = Chem.MolToSmiles(mol,canonical=True)
            # check if repeated
            if child.value in smiles_this_gen:
                repeat_num +=1
            else:
                smiles_this_gen.append(child.value)
                offspring.append(child)
    return offspring, failed_num,repeated_num
    
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

