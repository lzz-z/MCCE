import re
def extract_symbol_and_number(requirement_str):
    pattern = r'([<>]=?|==|!=)\s*(-?\d+(\.\d+)?)'
    match = re.search(pattern, requirement_str.strip())
    
    if match:
        symbol = match.group(1)  # Extracts the comparison symbol (e.g., '>=', '==', '!=', etc.)
        number = float(match.group(2))  # Extracts the number (as a float, e.g., 2, -3, 3.14, etc.)
        return symbol, number
    else:
        raise NotImplementedError('The format of requirement is wrong:', requirement_str)


def judge(requirement,input_mol_value,output_mol_value):
    metas = requirement.split(',')
    if metas[0] == 'the same':
        return (input_mol_value == output_mol_value)
    if metas[0] == 'equal':
        return (output_mol_value == float(metas[1]))
    if metas[0] == 'towards':
        towards_value = float(metas[1])
        return (abs(output_mol_value-towards_value)<abs(input_mol_value-towards_value))
    if len(metas) < 3 and metas[0] in ['increase','decrease']: # examples:  'increase, >=2' ,'decrease, >=2'
        direction = metas[0]
        if len(metas) == 1:
            symbol,number = '>',0
        elif len(metas) == 2:
            symbol,number =  extract_symbol_and_number(metas[1])
        if direction == 'increase':
            diff = output_mol_value - input_mol_value
        elif direction == 'decrease':
            diff = input_mol_value - output_mol_value
        else:
            raise NotImplementedError('Not implemented requirement:',requirement)
        if symbol == '>=':
            return diff >= float(number)
        elif symbol == '>':
            return diff > float(number)
        elif symbol == '<=':
            return diff <= float(number)
        elif symbol == '<':
            return diff < float(number)
        elif symbol == '==':
            return diff == float(number)
        else:
            raise NotImplementedError('Not implemented symbol:',symbol)
    elif len(metas) == 3:
        assert metas[0] == 'range' # only support range currently, e.g. 'range, 7, 8'
        a, b = [float(x) for x in requirement.split(',')[1:]]
        return (a<=output_mol_value<=b)

import re
from tqdm import tqdm
import requests
import json
import numpy as np
url = 'http://cpu1.ms.wyue.site:8000/process'
import time


def get_evaluation(evaluate_metric, smiles):
    data = {
        "ops": evaluate_metric,
        "data":smiles
    }
    response = requests.post(url, json=data)
    result = response.json()['results']
    return result

def extract_smiles_from_string(text):
    pattern = r"<candidate>(.*?)</candidate>"
    smiles_list = re.findall(pattern, text,flags=re.DOTALL)
    return smiles_list

def mean_sr(r,num_candiate=5):
    k = r.clip(0,num_candiate)
    new_sr = k.sum()/ (len(k)*num_candiate)
    return r.mean(), (r>0).sum()/len(r),new_sr
from algorithm.base import Item
def eval_mo_results(dataset,obj,ops=['qed','logp','donor']):
    hist_success_times = []
    prompts = dataset['prompts']
    requs = dataset['requirements']
    for index in tqdm(range(len(obj['final_pops']))):
        prompt = prompts[index]
        mol = extract_smiles_from_string(prompt)[0]
        #print(mol)
        final_pops = obj['final_pops'][index]
        input_mol = obj['init_pops'][index][-1] # get the original mol
        assert input_mol.value == mol
        #combine_mols = [[mol, i.value] for i in final_pops] # this format for compatibility, but properties like qed,logp will only
                                                            # evaluate the mols at [:,1]
        #eval_res = get_evaluation(ops,combine_mols)
        success_times = 0
        for output_mol in final_pops:
            if eval_one(ops,requs[index],input_mol,output_mol):
                success_times+=1
        #print('success times:',success_times)
        hist_success_times.append(success_times)
    return np.array(hist_success_times)

def eval_one(ops,requs,input_mol,output_mol):
    for op in ops:
        if op=='similarity':
            if not judge('range, 0.4, 1', input_mol.property[op],output_mol.property[op]):
                return False
        else:
            if not judge(requs[op+'_requ']['requirement'], input_mol.property[op],output_mol.property[op]):
                return False
    return True

