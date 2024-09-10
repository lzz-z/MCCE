import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import re
import copy
from tqdm import tqdm
# Set your OpenAI API key
import random
import torch
from functools import partial
import os
from algorithm.base import Item,History_Buffer
from openai import AzureOpenAI
from tdc.generation import MolGen

from eval import get_evaluation
import time
from model.util import nsga2_selection,so_selection

def generate_sentence(original_mol,requirements,properties):
    sentences = []
    for property in properties:
        value = requirements[property+'_requ']
        property_name = value["property"]
        source_smiles = value["source_smiles"]
        requirement = value["requirement"]

        # Check for specific requirement patterns directly using symbols
        if "increase" in requirement:
            if ">=" in requirement:
                threshold = requirement.split(">=")[-1].strip()
                sentence = f"help me increase the {property_name} value by at least {threshold}."
            elif ">" in requirement:
                threshold = requirement.split(">")[-1].strip()
                sentence = f"help me increase the {property_name} value to more than {threshold}."
            else:
                sentence = f"help me increase the {property_name} value."

        elif "decrease" in requirement:
            if "<=" in requirement:
                threshold = requirement.split("<=")[-1].strip()
                sentence = f"help me decrease the {property_name} value to at most {threshold}."
            elif "<" in requirement:
                threshold = requirement.split("<")[-1].strip()
                sentence = f"help me decrease the {property_name} value to less than {threshold}."
            else:
                sentence = f"help me decrease the {property_name} value."

        elif "range" in requirement:
            # Extract the range values from the string
            range_values = requirement.split(",")[1:]
            range_start = range_values[0].strip()
            range_end = range_values[1].strip()
            sentence = f"help me keep the {property_name} value within the range {range_start} to {range_end}."

        elif "the same" in requirement:
            sentence = f"help me keep the {property_name} value the same."

        elif any(op in requirement for op in [">=", "<=", "=", ">", "<"]):
            # Directly use the symbols for constraints
            sentence = f"help me ensure the {property_name} value is {requirement}."
            
        else:
            sentence = f"help me modify the {property_name} value."
        sentences.append(sentence)
    sentences = f'Suggest new molecules based on molecule <mol>{original_mol}</mol> and ' + 'and '.join(sentences) 
    return sentences

# Example metadata
metadata = {
    "qed_requ": {
        "source_smiles": "O=C([C@H]1CCCC[C@H]1N1CCCC1=O)N1CC2(CC(F)C2)C1",
        "reference_smiles": "NNC(=O)C(=O)NC1CC2(C1)CN(C(=O)[C@H]1CCCC[C@H]1N1CCCC1=O)C2",
        "property": "QED",
        "requirement": "decrease <=2"
    },
    "logp_requ": {
        "source_smiles": "CCCCC(CC)COCOc1ccc([C@@H](O)C(=O)N[C@@H]2[C@H]3COC[C@@H]2CN(C(=O)CCc2ccccc2Cl)C3)cc1",
        "reference_smiles": "COc1ccc([C@@H](O)C(=O)N[C@H]2[C@@H]3COC[C@H]2CN(C(=O)CCc2ccccc2Cl)C3)cc1",
        "property": "logP",
        "requirement": "range, 2, 3"
    },
    "donor_requ": {
        "source_smiles": "O=C(NC[C@H]1CCOc2ccccc21)c1ccc(F)c(C(F)(F)F)c1",
        "reference_smiles": "CC(C)C[NH+](CC(=O)[O-])C(F)(F)c1cc(C(=O)NC[C@H]2CCOc3ccccc32)ccc1F",
        "property": "donor",
        "requirement": "increase >= 1"
    },
}

# Generate sentences based on metadata
generate_sentence('CCH',metadata,['qed','donor'])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_smiles_from_string(text):
    pattern = r"<mol>(.*?)</mol>"
    smiles_list = re.findall(pattern, text)
    return smiles_list

def split_list(lst, n):
    """Splits the list lst into n nearly equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

class MOO:
    def __init__(self, reward_system, llm,property_list,config):
        self.reward_system = reward_system
        self.config = config
        self.llm = llm
        self.history = History_Buffer()
        self.property_list = property_list
        self.moles_df = None
        self.pop_size = self.config.get('optimization.pop_size')
        
    def init_mol_dataset(self):
        print('Loading ZINC dataset...')
        data = MolGen(name='ZINC')
        split = data.get_split()
        self.moles_df = split['train']

    def generate_initial_population(self, mol1, n):
        num_blocks = 200
        combs = [[mol1, mol2] for mol2 in self.moles_df.smiles]
        combs_blocks = split_list(combs, num_blocks)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(get_evaluation, ['similarity'], block) for block in combs_blocks]
            results = [future.result() for future in futures]

        combined_results = []
        for result in results:
            combined_results.extend(result['similarity'])

        self.moles_df['similarity'] = combined_results
        top_n = self.moles_df.nlargest(n - 1, 'similarity').smiles.values.tolist()
        top_n.append(mol1)
        return [Item(i,self.property_list) for i in top_n]

    def crossover(self, parent_list):
        pop_content = self._prepare_pop_content(parent_list)
        prompt = generate_sentence(self.original_mol,self.requirement_meta,self.property_list)

        prompt = (
        prompt + 
        "I have some molecules with their objective values. "
        + pop_content +
        " Give me two new molecules that are different from all points above, and not dominated by any of the above. "
        "You can do it by applying crossover on the points I give to you. "
        f"Please note when you try to achieving these objectives, the molecules you propose should be similar to the original molecule <mol>{self.original_mol}</mol>. "
        "Do not write code. Do not give any explanation. Each output new molecule must start with <mol> and end with </mol> in SIMLE form"
        )
        print(prompt)
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        return [Item(smile,self.property_list) for smile in new_smiles],prompt,response

    def _prepare_pop_content(self, ind_list):
        pop_content = ""
        for ind in ind_list:
            pop_content += f"<mol>{ind.value}</mol>,"
            for index,property in enumerate(ind.property_list):
                pop_content += f"{property}:{ind.raw_scores[index]},  "
            pop_content += '\n'
        return pop_content

    def evaluate(self, smiles_list):
        ops = self.property_list
        res = self.reward_system.evaluate(ops,smiles_list)
        results = np.zeros([len(smiles_list),len(ops)])
        raw_results = np.zeros([len(smiles_list),len(ops)])
        for i,op in enumerate(ops):
            part_res = res[op]
            for j,inst in enumerate(part_res):
                inst = inst[0]
                raw_results[j,i] = inst
                if op=='qed':
                    if 'increase' in self.requirement_meta['qed_requ']['requirement']:
                        results[j,i] = -inst
                    else:
                        results[j,i] = inst
                elif op=='logp':
                    logp_requ = self.requirement_meta['logp_requ']['requirement']
                    if 'increase' in logp_requ:
                        results[j,i] = -inst / 10
                    elif 'range' in logp_requ:
                        a, b = [int(x) for x in logp_requ.split(',')[1:]]
                        mid = (b+a)/2
                        results[j,i] = np.clip(abs(inst-mid) * 1/ ( (b-a)/2), a_min=0,a_max=1)    # 2-3    2.5  3-2.5=0.5 * 2     1 0
                        #if a<=inst<=b:
                        #    results[j,i] = 0
                        #else:
                        #    results[j,i] = 1     
                    else:
                        results[j,i] = inst/10
                elif op=='donor':
                    #print('donor num',inst,donor_num,j,i)
                    donor_requ = self.requirement_meta['donor_requ']['requirement']
                    donor_num = self.requirement_meta['donor_num']
                    if donor_requ == 'increase' and inst - donor_num>0:
                        results[j,i] = 0
                    elif donor_requ == 'decrease' and donor_num - inst>0:
                        results[j,i] = 0
                    elif donor_requ == 'same' and donor_num == inst:
                        results[j,i] = 0
                    elif donor_requ == 'increase, >=2' and inst - donor_num>=2:
                        results[j,i] = 0
                    else:
                        results[j,i] = 1  
                else:
                    raise NotImplementedError
        return results,raw_results

    def evaluate_all(self,items):
        smiles_list = [i.value for i in items]
        smiles_list = [[i,smiles_list[0]] for i in smiles_list]
        fitnesses,raw_results = self.evaluate(smiles_list)
        for i,ind in enumerate(items):
            ind.scores = fitnesses[i]
            ind.raw_scores = raw_results[i]

    def run(self, prompt,requirements):
        """High level logic"""
        self.requirement_meta = requirements
        ngen= self.config.get('optimization.ngen')
        

        self.init_mol_dataset()
        
        #initialization 
        mol = extract_smiles_from_string(prompt)[0]
        self.original_mol = mol

        population = self.generate_initial_population(mol1=mol, n=self.pop_size)
        donor_num = get_evaluation(['donor'], [[mol, mol]])['donor'][0][0]
        self.requirement_meta['donor_num'] = donor_num
        self.evaluate_all(population)

        init_pops = copy.deepcopy(population)

        #offspring_times = self.config.get('optimization.eval_budge') // ngen //2
        offspring_times = self.pop_size //2
        for gen in tqdm(range(ngen)):
            offspring = self.generate_offspring(population, prompt, offspring_times)
            population = self.select_next_population(population, offspring, self.pop_size)
        return init_pops,population

    def generate_offspring(self, population, prompt, offspring_times):
        offspring = []
        #for _ in range(offspring_times): # 20 10 crossver+mutation 20 
        parents = [random.sample(population, 2) for i in range(offspring_times)]
        while True:
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.crossover, parent_list=parent_list) for parent_list in parents]
                    results = [future.result() for future in futures]
                    children, prompts, responses = zip(*results) #[[item,item],[item,item]] # ['who are you value 1', 'who are you value 2'] # ['yes, 'no']
                    break
            except Exception as e:
                print('retry in 60s, exception ',e)
                time.sleep(90)
        #parents = random.sample(population, 2)
        #children,prompt,response = self.crossover(prompt, parents) 
        for child_pair in children:
            self.evaluate_all(child_pair)
            offspring.extend(child_pair)
        self.history.push(prompts,children,responses) 
        return offspring

    def select_next_population(self, population, offspring, pop_size):
        combined_population = population + offspring
        if len(self.property_list)>1:
            return nsga2_selection(combined_population, pop_size)
        else:
            return so_selection(combined_population, pop_size)
c,p,r = None,None,None