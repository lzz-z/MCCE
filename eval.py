def judge_donor(  requirement, simi_requirement, similarity, donor_input_mol, donor_response_mol, delta= 1e-9 ):
    if requirement == 'increase':
        return (donor_response_mol > donor_input_mol + delta and similarity>simi_requirement)
    if requirement == 'decrease':
        return (donor_response_mol < donor_input_mol - delta  and similarity>simi_requirement)
    if requirement == 'the same':        
        return (abs(donor_input_mol - donor_response_mol) < delta and similarity>simi_requirement)
    if requirement == 'increase, >=2':
        return (donor_response_mol - donor_input_mol >= 2 and similarity>simi_requirement)
    if requirement == 'decrease, >=2':
        return (donor_input_mol - donor_response_mol>=2 and similarity>simi_requirement)
        
    raise ValueError(f'Invalid requirement: {requirement}')

def judge_qed(  requirement, simi_requirement, similarity, qed_input_mol, qed_response_mol, delta=1e-9 ):
    if abs(qed_response_mol) == 0:
        return False
    if requirement == 'increase':
        return (qed_response_mol > qed_input_mol + delta and similarity>simi_requirement)
    if requirement == 'decrease':
        return (qed_response_mol < qed_input_mol - delta and similarity>simi_requirement)
    if requirement == 'increase, >=0.1':
        return ( qed_response_mol - qed_input_mol >= 0.1 and similarity>simi_requirement)
    if requirement == 'decrease, >=0.1':
        return ( qed_input_mol - qed_response_mol >= 0.1 and similarity>simi_requirement)
    raise ValueError(f'Invalid requirement: {requirement}')

def judge_logp(  requirement, simi_requirement, similarity, logp_input_mol, logp_response_mol, delta=1e-9 ):
    if abs(logp_response_mol) == 100:
        return False
    if requirement == 'increase':
        return (logp_response_mol > logp_input_mol + delta and similarity>simi_requirement)
    if requirement == 'decrease':
        return (logp_response_mol < logp_input_mol - delta and similarity>simi_requirement)
    if 'range,' in requirement:
        a, b = [int(x) for x in requirement.split(',')[1:]]
        return (a<=logp_response_mol<=b and similarity>simi_requirement)
    raise ValueError(f'Invalid requirement: {requirement}')


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
    pattern = r"<mol>(.*?)</mol>"
    smiles_list = re.findall(pattern, text)
    return smiles_list

def mean_sr(r,num_candiate=5):
    k = r.clip(0,num_candiate)
    new_sr = k.sum()/ (len(k)*num_candiate)
    return r.mean(), (r>0).sum()/len(r),new_sr

def eval_mo_results(dataset,obj,similarity_requ=0.4,ops=['qed','logp','donor'],candidate_num=20):
    hist_success_times = []
    prompts = dataset['prompts']
    requs = dataset['requirements']
    for index in tqdm(range(len(obj['final_pops']))):
        prompt = prompts[index]
        mol = extract_smiles_from_string(prompt)[0]
        #print(mol)
        final_pops = obj['final_pops'][index]
        combine_mols = [[mol, i.value] for i in final_pops]
        eval_res = get_evaluation(['similarity']+ops,combine_mols)
        success_times = 0
        for i in range(candidate_num):
            if eval_one(ops,requs,index,similarity_requ,eval_res,i):
                success_times+=1
        #print('success times:',success_times)
        hist_success_times.append(success_times)
        
    return np.array(hist_success_times)

def eval_one(ops,requs,index,similarity_requ,eval_res,i):
    if 'donor' in ops:
        input_mol_donor = eval_res['donor'][0][0]
    if 'qed' in ops:
        input_mol_qed = eval_res['qed'][0][0]
    if 'logp' in ops:
        input_mol_logp = eval_res['logp'][0][0]
    for op in ops:
        if op == 'qed' and not judge_qed(requs[index]['qed_requ']['requirement'],similarity_requ,eval_res['similarity'][i],input_mol_qed,eval_res['qed'][i][1]):
            return False
        if op=='logp' and not judge_logp(requs[index]['logp_requ']['requirement'],similarity_requ,eval_res['similarity'][i],input_mol_logp,eval_res['logp'][i][1]):
            return False
        if op=='donor' and not judge_donor(requs[index]['donor_requ']['requirement'],similarity_requ,eval_res['similarity'][i],input_mol_donor,eval_res['donor'][i][1]):
            return False
    return True