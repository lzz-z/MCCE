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
from algorithm.base import Item,HistoryBuffer
from openai import AzureOpenAI
from tdc.generation import MolGen
from rdkit.Chem import AllChem
import json
from eval import get_evaluation
import time
from model.util import nsga2_selection,so_selection
from algorithm import PromptTemplate
from eval import judge
import pickle
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
    def __init__(self, reward_system, llm,property_list,config,seed):
        self.reward_system = reward_system
        self.config = config
        self.seed = seed
        self.llm = llm
        self.history = HistoryBuffer()
        self.property_list = property_list
        self.moles_df = None
        self.pop_size = self.config.get('optimization.pop_size')
        self.budget = self.config.get('optimization.eval_budget')
        self.init_mol_dataset()
        self.prompt_module = getattr(PromptTemplate ,self.config.get('model.prompt_module',default='Prompt'))
        self.successful_moles = []
        self.failed_moles = []
        self.history_moles = []
        self.all_mols = []
        self.results_dict = []
        self.repeat_num = 0
        self.failed_num = 0
        self.llm_calls = 0

    def init_mol_dataset(self):
        print('Loading ZINC dataset...')
        data = MolGen(name='ZINC')
        self.moles_df = data.get_data()

    def generate_initial_population(self, mol1, n):
        with open('/home/v-nianran/src/MOLLEO/multi_objective/ini_smiles','r') as f:
            a = f.readlines()
        a = [i.replace('\n','') for i in a]
        return [Item(i,self.property_list) for i in a]


        '''
        filepath = '/home/v-nianran/src/MOLLM/data/zinc250_5goals.pkl'
        with open(filepath, 'rb') as f:
            all_mols_zinc = pickle.load(f)
        print(f"init pop loaded from to {filepath}")
        # return all_mols_zinc['worst500'][-100:]
        return all_mols_zinc['best500'][:100]'''

        top_n = self.moles_df.sample(n - 1).smiles.values.tolist()
        top_n.append(mol1)
        return [Item(i,self.property_list) for i in top_n]

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

    def mutation(self, parent_list):
        prompt = self.prompt_generator.get_mutation_prompt(parent_list,self.history_moles)
        #print(prompt,'\n\n')
        #assert False
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        return [Item(smile,self.property_list) for smile in new_smiles],prompt,response

    def crossover(self, parent_list):
        prompt = self.prompt_generator.get_crossover_prompt(parent_list,self.history_moles)
        #print(prompt,'\n\n')
        #assert False
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        return [Item(smile,self.property_list) for smile in new_smiles],prompt,response

    def evaluate(self, smiles_list):
        ops = self.property_list
        res = self.reward_system.evaluate(ops,smiles_list)
        results = np.zeros([len(smiles_list),len(ops)])
        raw_results = np.zeros([len(smiles_list),len(ops)])
        for i,op in enumerate(ops):
            part_res = res[op]
            #print('part res',part_res)
            for j,inst in enumerate(part_res):
                if isinstance(inst,list):
                    inst = inst[1]
                raw_results[j,i] = inst
                value = self.transform4moo(inst,op,)
                results[j,i] = value
        return results,raw_results

    def transform4moo(self,value,op):
        '''
         this means when the requirement is satisfied, if will give 0 (optimal value), other 1, this means when the requirement is satified,
         this objective will not be main objectives to optimizes, this is useful when we only want the object to reach a certain 
         threshold instead of maximizing or minimizing it
        '''
        original_value = self.original_mol.property[op]
        requirement = self.requirement_meta[f'{op}_requ']['requirement']
        if op =='similarity':
            return -value
        if op == 'reduction_potential':
            towards_value = float(requirement.split(',')[1])
            return abs(value - towards_value)/5
        if op in ['donor','smarts_filter']: 
            is_true = judge(requirement,original_value,value)
            if is_true:
                return 0
            else:
                return 1
        else:
            '''
            this means the transformed value will only be minimized to as low as possible
            '''
            if 'range' in requirement:
                a, b = [float(x) for x in requirement.split(',')[1:]]
                mid = (b+a)/2
                return np.clip(abs(value-mid) * 1/ ( (b-a)/2), a_min=0,a_max=1)
            if op in ['logp','logs','sa']:
                value = value/10
            if 'increase' in requirement:
                return -value
            elif 'decrease' in requirement:
                return value
            else:
                print('only support increase or decrease and range for minimizing')
                raise NotImplementedError

    def evaluate_all(self,items):
        smiles_list = [i.value for i in items]
        smiles_list = [[self.original_mol.value,i] for i in smiles_list]
        fitnesses,raw_results = self.evaluate(smiles_list)
        for i,ind in enumerate(items):
            ind.scores = fitnesses[i]
            ind.assign_raw_scores(raw_results[i])
            #ind.raw_scores = raw_results[i]

    def store_history_moles(self,pops):
        for i in pops:
            if i.value not in self.history_moles:
                self.history_moles.append(i.value)
            self.all_mols.append(i)

    def log(self):
        top100 = sorted(self.all_mols, key=lambda item: item.total, reverse=True)[:100]
        top10 = top100[:10]
        avg_top10 = np.mean([i.total for i in top10])
        avg_top100 = np.mean([i.total for i in top100])
        avg_sa = np.mean([i.property['sa'] for i in top100])
        diversity_top100 = self.reward_system.all_evaluators['diversity']([i.value for i in top100])

        already = 0
        all_zinc_mols = self.moles_df.smiles.values
        for i in self.all_mols:
            if i.value in all_zinc_mols:
                already += 1 
        self.results_dict.append(
            {   'all_unique_moles': len(self.history_moles),
                'llm_calls': self.llm_calls,
                'Uniqueness':1-self.repeat_num/(self.llm_calls*2+1e-6),
                'Validity':1-self.failed_num/(self.llm_calls*2+1e-6),
                'Novelty':1-already/(self.llm_calls*2+1e-6),
                'avg_top1':top10[0].total,
                'avg_top10':avg_top10,
                'avg_top100':avg_top100,
                'avg_sa':avg_sa,
                'div':diversity_top100,
            })
        json_path = os.path.join(self.config.get('save_dir'),'_'.join(self.property_list) + '_' + self.config.get('save_suffix') + f'_{self.seed}'+'.json')
        with open(json_path,'w') as f:
            json.dump(self.results_dict, f, indent=4)
        print(f'{len(self.history_moles)}/{self.budget} | '
                f'Uniqueness:{1-self.repeat_num/(self.llm_calls*2+1e-6)} | '
                f'Validity:{1-self.failed_num/(self.llm_calls*2+1e-6)} | '
                f'Novelty:{1-already/(self.llm_calls*2+1e-6)} | '
                f'llm_calls: {self.llm_calls} | '
                f'avg_top1: {top10[0].total:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f} | '
                f'avg_sa: {avg_sa:.3f} | '
                f'div: {diversity_top100:.3f}')

    def generate_experience(self):
        if np.random.random()>0.5:
            self.prompt_generator.experience = None
            print('no experience this generation')
        else:
            prompt,best_moles_prompt = self.prompt_generator.make_experience_prompt(self.all_mols)
            response = self.llm.chat(prompt)
            self.prompt_generator.experience= best_moles_prompt + "\n I have some experience of proposing such best molecules, you can take advantage of my best molecules and experience: \n" + response + "\n"
            print('length exp:',len(self.prompt_generator.experience))

    def run(self, prompt,requirements):
        """High level logic"""
        set_seed(self.seed)
        start_time = time.time()
        self.requirement_meta = requirements
        ngen= self.config.get('optimization.ngen')
        
        #initialization 
        mol = extract_smiles_from_string(prompt)[0]

        population = self.generate_initial_population(mol1=mol, n=self.pop_size)
        self.store_history_moles(population)
        self.original_mol = population[-1] # this original_mol does not have property
        self.evaluate_all(population)
        self.original_mol = population[-1] # this original_mol has property
        self.log()
        self.prompt_generator = self.prompt_module(self.original_mol,self.requirement_meta,self.property_list)
        init_pops = copy.deepcopy(population)

        #offspring_times = self.config.get('optimization.eval_budge') // ngen //2
        offspring_times = self.pop_size //2
        #for gen in tqdm(range(ngen)):
        while True:
            offspring = self.generate_offspring(population, offspring_times)
            population = self.select_next_population(population, offspring, self.pop_size)
            self.log()
            #self.generate_experience()
            if len(self.all_mols) >= self.budget:
                break
        print(f'=======> total running time { (time.time()-start_time)/3600 :.2f} hours <=======')
        store_path = os.path.join(self.config.get('save_dir'),'_'.join(self.property_list) + '_' + self.config.get('save_suffix') + f'_{self.seed}' +'.pkl')
        data = {
            'history':self.history,
            'init_pops':init_pops,
            'final_pops':population,
            'all_mols':self.all_mols,
            'properties':self.property_list,
            'evaluation': self.results_dict,
            'running_time':f'{(time.time()-start_time)/3600:.2f} hours'
        }
        with open(store_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {store_path}")
        return init_pops,population  # 计算效率

    def generate_offspring(self, population, offspring_times):
        #for _ in range(offspring_times): # 20 10 crossver+mutation 20 
        parents = [random.sample(population, 2) for i in range(offspring_times)]
        while True:
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = []
                    for parent_list in parents:
                        if np.random.random()<1.5:
                            futures.append(executor.submit(self.crossover, parent_list=parent_list))
                        else:
                            futures.append(executor.submit(self.mutation, parent_list=parent_list))
                    #futures = [executor.submit(self.crossover, parent_list=parent_list) for parent_list in parents]
                    results = [future.result() for future in futures]
                    children, prompts, responses = zip(*results) #[[item,item],[item,item]] # ['who are you value 1', 'who are you value 2'] # ['yes, 'no']
                    self.llm_calls += len(results)
                    break
            except Exception as e:
                print('retry in 60s, exception ',e)
                time.sleep(90)
        tmp_offspring = []
        smiles_this_gen = []
        for child_pair in children:
            for child in child_pair:
                if child.value in smiles_this_gen:
                    self.repeat_num += 1
                else:
                    tmp_offspring.append(child)
                    smiles_this_gen.append(child.value)
        # check if the child is valid
        offspring = self.check_valid(tmp_offspring)
        if len(offspring) == 0:
            return []
        self.evaluate_all(offspring)
        self.store_history_moles(offspring)
        self.history.push(prompts,children,responses) 
        return offspring

    def check_valid(self,children):
        # may use Chem.MolFromSmiles() to check validity
        offspring = []
        for child in children:
            if self.is_valid(child):
                offspring.append(child)
        return offspring
    
    def is_valid(self,child):
        mol = AllChem.MolFromSmiles(child.value)
        if mol is None:
            self.failed_num += 1
            return False
        if child.value in self.history_moles:
            self.repeat_num +=1
            return False
        return True

    def select_next_population(self, population, offspring, pop_size):
        combined_population = offspring + population 

        if len(self.property_list)>1:
            return nsga2_selection(combined_population, pop_size)
        else:
            return so_selection(combined_population, pop_size)
    
    def no_selection(self,combined_population, pop_size):
        return combined_population[:pop_size] 
c,p,r = None,None,None