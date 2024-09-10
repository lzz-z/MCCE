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

import numpy as np
from tqdm import tqdm
import requests
import json

# 定义 API 端点 URL
url = 'http://cpu1.ms.wyue.site:8000/process'


def get_evaluation(evaluate_metric, smiles):
    data = {
        "ops": evaluate_metric,
        "data":smiles
    }
    response = requests.post(url, json=data)
    result = response.json()['results']
    return result

def mean_sr(r):
    return r.mean(), (r>0).sum()/len(r)
def eval_mo_results(df,similarity_requ=0.4,length=99999,ops=['qed','logp','donor'],qed_meta=None,logp_meta=None,donor_meta=None,candidate_num=20):
    #key = 'mo_gpt4_direct_wo_history'
    #similarity_requ = 0.4
    hist_success_times = []
    for index,row in tqdm(df[:length].iterrows()):
        mol = row['input_mol']
        #print(mol)
        combine_mols = [[mol,row[f'response{i+1}']] for i in range(candidate_num)]
        eval_res = get_evaluation(['similarity']+ops,combine_mols)
        #print(eval_res)
        if 'donor' in ops:
            input_mol_donor = eval_res['donor'][0][0]
        if 'qed' in ops:
            input_mol_qed = eval_res['qed'][0][0]
        if 'logp' in ops:
            input_mol_logp = eval_res['logp'][0][0]
        success_times = 0
        for i in range(candidate_num):
            if len(ops) == 3:
                if judge_donor(donor_meta[index]['requirement'],similarity_requ,eval_res['similarity'][i],input_mol_donor,eval_res['donor'][i][1]) and \
                    judge_logp(logp_meta[index]['requirement'],similarity_requ,eval_res['similarity'][i],input_mol_logp,eval_res['logp'][i][1]) and \
                    judge_qed(qed_meta[index]['requirement'],similarity_requ,eval_res['similarity'][i],input_mol_qed,eval_res['qed'][i][1]):
                    success_times += 1
            elif len(ops)==1 and ops[0]=='qed' and judge_qed(qed_meta[index]['requirement'],similarity_requ,eval_res['similarity'][i],input_mol_qed,eval_res['qed'][i][1]):
                success_times+=1
            elif len(ops)==1 and ops[0]=='logp' and judge_logp(logp_meta[index]['requirement'],similarity_requ,eval_res['similarity'][i],input_mol_logp,eval_res['logp'][i][1]):
                success_times+=1
            elif len(ops)==1 and ops[0]=='donor' and judge_donor(donor_meta[index]['requirement'],similarity_requ,eval_res['similarity'][i],input_mol_donor,eval_res['donor'][i][1]):
                success_times+=1
        #print('success times:',success_times)
        hist_success_times.append(success_times)
    return np.array(hist_success_times)
