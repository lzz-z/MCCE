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
from rdkit import Chem
import json
from eval import get_evaluation
import time
from model.util import nsga2_so_selection
from algorithm import PromptTemplate
from eval import judge
import pygmo as pg
import pickle

from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
def cal_hv(scores):
    ref_point = np.array([1.1]*len(scores[0]))
    hv = HV(ref_point=ref_point)
    nds = NonDominatedSorting().do(scores,only_non_dominated_front=True)
    scores = scores[nds]
    return hv(scores)

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

def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer, key=lambda kv: kv[1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[0].total, reverse=True))[:top_n]
        top_n_now = np.mean([item[0].total for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[0].total, reverse=True))[:top_n]
    top_n_now = np.mean([item[0].total for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls

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
        self.save_dir = os.path.join(self.config.get('save_dir'),self.config.get('model.name'))
        self.init_mol_dataset()
        self.prompt_module = getattr(PromptTemplate ,self.config.get('model.prompt_module',default='Prompt'))
        self.history_moles = []
        self.all_mols = []
        self.mol_buffer = [] # same as all_mols but with orders for computing auc
        self.results_dict = {'results':[]}
        self.history_experience = []
        self.repeat_num = 0
        self.failed_num = 0
        self.generated_num = 0
        self.llm_calls = 0
        self.patience = 0
        self.old_score = 0
        self.early_stopping = False
    def init_mol_dataset(self):
        print('Loading ZINC dataset...')
        data = MolGen(name='ZINC')
        self.moles_df = data.get_data()

    def generate_initial_population(self, mol1, n):
        '''
        with open('/home/v-nianran/src/MOLLEO/multi_objective/ini_smiles','r') as f:
            a = f.readlines()
        a = [i.replace('\n','') for i in a]
        return [Item(i,self.property_list) for i in a]
        '''
        '''
        with open('/home/v-nianran/src/MOLLM/data/scaffold_smiles.txt','r') as f:
            smiles = f.readlines()
        smiles = [smile.replace('\n','') for smile in smiles]
        top_n = self.moles_df.sample(100 - len(smiles)).smiles.values.tolist()
        smiles.extend(top_n)
        print('load scaffold smiles')
        return [Item(i,self.property_list) for i in smiles]
        '''
        with open('/home/hp/src/MOLLM/data/data_goal5.json','r') as f:
            data = json.load(f)
        data_type = self.config.get('initial_pop')
        print(f'loading {data_type} as initial pop!')
        smiles = data[data_type]
        return [Item(i,self.property_list) for i in smiles]
        
        filepath = '/home/v-nianran/src/MOLLM/data/zinc250_5goals.pkl'
        with open(filepath, 'rb') as f:
            all_mols_zinc = pickle.load(f)
        print(f"init pop loaded from to {filepath}")
        # return all_mols_zinc['worst500'][-100:]
        return all_mols_zinc['best500'][:100]

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
        use_experience = False
        if len(self.history_moles)>= self.budget // 2:
            use_experience = True
            
        prompt = self.prompt_generator.get_prompt('mutation',parent_list,self.history_moles,use_experience=use_experience)
        #print('mutation prompt \n\n',prompt)
        
        response = self.llm.chat(prompt)
        #print('response:',response,'\n\n\n')
        #assert False
        new_smiles = extract_smiles_from_string(response)
        return [Item(smile,self.property_list) for smile in new_smiles],prompt,response

    def crossover(self, parent_list):
        use_experience = False
        if len(self.history_moles)>= self.budget // 2:
            use_experience = True
        prompt = self.prompt_generator.get_prompt('crossover',parent_list,self.history_moles,use_experience=use_experience)
        #print('crossover prompt',prompt)
        
        response = self.llm.chat(prompt)
        #print('response:',response,'\n\n\n')
        #assert False
        
        new_smiles = extract_smiles_from_string(response)
        return [Item(smile,self.property_list) for smile in new_smiles],prompt,response
    
    def explore(self, parent_list):
        prompt = self.prompt_generator.get_prompt('explore',parent_list,self.all_mols)
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        #print('response:',response)
        #print('smiles',new_smiles)
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
            return abs(value - towards_value)/5*2
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
                return 1-value
            elif 'decrease' in requirement:
                return value
            else:
                raise NotImplementedError('only support increase or decrease and range for minimizing')

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
            self.mol_buffer.append([i, len(self.mol_buffer)+1])
            self.all_mols.append(i)

    def log(self,finish=False):
        auc1 = top_auc(self.mol_buffer, 1, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc10 = top_auc(self.mol_buffer, 10, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc100 = top_auc(self.mol_buffer, 100, finish=finish, freq_log=100, max_oracle_calls=self.budget)


        top100 = sorted(self.all_mols, key=lambda item: item.total, reverse=True)[:100]
        top10 = top100[:10]
        avg_top10 = np.mean([i.total for i in top10])
        avg_top100 = np.mean([i.total for i in top100])
        
        diversity_top100 = self.reward_system.all_evaluators['diversity']([i.value for i in top100])

        scores = np.array([i.scores for i in top100])
        volume = cal_hv(scores)

        new_score = avg_top100
        # import ipdb; ipdb.set_trace()
        if (new_score - self.old_score) < 5e-5:
            self.patience += 1
            if self.patience >= 5:
                print('convergence criteria met, abort ...... ')
                self.early_stopping = True
        else:
            self.patience = 0
        self.old_score = new_score
        
        if 'bbbp' in self.property_list:
            top1_bbbp = top10[0].property['bbbp']
            top10_bbbp = np.mean([i.property['bbbp'] for i in top10])
            top100_bbbp = np.mean([i.property['bbbp'] for i in top100])
            
            # already = 0
            #all_zinc_mols = self.moles_df.smiles.values
            #for i in self.all_mols:
            #    if i.value in all_zinc_mols:
            #        already += 1 
            self.results_dict['results'].append(
                {   'all_unique_moles': len(self.history_moles),
                    'llm_calls': self.llm_calls,
                    'Uniqueness':1-self.repeat_num/(self.generated_num+1e-6),
                    'Validity':1-self.failed_num/(self.generated_num+1e-6),
                    #'Novelty':1-already/(self.generated_num+1e-6),
                    'avg_top1':top10[0].total,
                    'avg_top10':avg_top10,
                    'avg_top100':avg_top100,
                    'top1_auc':auc1,
                    'top10_auc':auc10,
                    'top100_auc':auc100,
                    'hypervolume':volume,
                    'bbbp_top1':top1_bbbp,
                    'bbbp_top10':top10_bbbp,
                    'bbbp_top100':top100_bbbp,
                    'div':diversity_top100,
                    'generated_num':self.generated_num
                })
            print(f'{len(self.history_moles)}/{self.budget} /all generated: {self.generated_num} | '
                f'Uniqueness:{1-self.repeat_num/(self.generated_num+1e-6):.4f} | '
                f'Validity:{1-self.failed_num/(self.generated_num+1e-6):.4f} | '
                #f'Novelty:{1-already/(self.generated_num+1e-6):.4f} | '
                f'llm_calls: {self.llm_calls} | '
                f'avg_top1: {top10[0].total:.4f} | '
                f'avg_top10: {avg_top10:.4f} | '
                f'avg_top100: {avg_top100:.4f} | '
                f'avg_top1 bbbp: {top1_bbbp:.4f} | '
                f'avg_top10 bbbp: {top10_bbbp:.4f} | '
                f'avg_top100 bbbp: {top100_bbbp:.4f} | '
                f'top1_auc : {auc1:.4f} | '
                f'top10_auc : {auc10:.4f} | '
                f'top100_auc : {auc100:.4f} | '
                f'hv: {volume:.4f} | '
                f'div: {diversity_top100:.4f}')
        else:
            self.results_dict['results'].append(
                {   'all_unique_moles': len(self.history_moles),
                    'llm_calls': self.llm_calls,
                    'Uniqueness':1-self.repeat_num/(self.generated_num+1e-6),
                    'Validity':1-self.failed_num/(self.generated_num+1e-6),
                    #'Novelty':1-already/(self.generated_num+1e-6),
                    'avg_top1':top10[0].total,
                    'avg_top10':avg_top10,
                    'avg_top100':avg_top100,
                    'top1_auc':auc1,
                    'top10_auc':auc10,
                    'top100_auc':auc100,
                    'hypervolume':volume,
                    'div':diversity_top100,
                    'generated_num':self.generated_num})
            print(f'{len(self.history_moles)}/{self.budget}/all generated: {self.generated_num} | '
                  f'len mol_buffer{len(self.mol_buffer)} | '
                f'Uniqueness:{1-self.repeat_num/(self.generated_num+1e-6):.4f} | '
                f'Validity:{1-self.failed_num/(self.generated_num+1e-6):.4f} | '
                #f'Novelty:{1-already/(self.generated_num+1e-6):.4f} | '
                f'llm_calls: {self.llm_calls} | '
                f'avg_top1: {top10[0].total:.4f} | '
                f'avg_top10: {avg_top10:.4f} | '
                f'avg_top100: {avg_top100:.4f} | '
                f'top1_auc : {auc1:.4f} | '
                f'top10_auc : {auc10:.4f} | '
                f'top100_auc : {auc100:.4f} | '
                f'hv: {volume:.4f} | '
                f'div: {diversity_top100:.4f}')
        
        json_path = os.path.join(self.save_dir,"results",'_'.join(self.property_list) + '_' + self.config.get('save_suffix') + f'_{self.seed}'+'.json')
        if not os.path.exists(os.path.dirname(json_path)):
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            
        self.results_dict['params'] = self.config.to_string()
        self.results_dict['history_experience']  = self.history_experience
        with open(json_path,'w') as f:
            json.dump(self.results_dict, f, indent=4)
        
        

    def update_experience(self):
        prompt,best_moles_prompt,bad_moles_prompt = self.prompt_generator.make_experience_prompt(self.all_mols)
        response = self.llm.chat(prompt)
        
        #self.prompt_generator.experience = (f"I already have some experience, take advantage of them :{response}"
        #                                    )
        self.prompt_generator.pure_experience = response
        self.prompt_generator.experience = (f"I already have some experience: <experience> {response} </experience>" 
                                            f"You can take advantage of these experience and try to propose better molecules according to the objectives.\n")
        #self.prompt_generator.experience = (f"I already have some experience and some good and bad moleculles, the experience is: <experience> {response}" 
        #                                    f"good example molecules are: {best_moles_prompt},\n"
        #                                    f"and bad example molecules that you need to avoid molecules like them: {bad_moles_prompt}. </experience>\n"
        #                                    f"You can take advantage of them and try to propose better molecules according to the objectives.\n"
        #                                    )
        
        self.history_experience.append(self.prompt_generator.experience) 
        print('length exp:',len(self.prompt_generator.experience))

    def run(self, prompt,requirements):
        """High level logic"""
        print('exper_name',self.config.get('exper_name'))
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
        self.prompt_generator = self.prompt_module(self.original_mol,self.requirement_meta,self.property_list,
                                                   self.config.get('model.experience_prob'))
        init_pops = copy.deepcopy(population)

        self.num_gen = 0
        #for gen in tqdm(range(ngen)):
        store_path = os.path.join(self.save_dir,'mols','_'.join(self.property_list) + '_' + self.config.get('save_suffix') + f'_{self.seed}' +'.pkl')
        if not os.path.exists(os.path.dirname(store_path)):
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
        while True:
            offspring_times = max(min(self.pop_size //2, (self.budget -len(self.all_mols)) //2),1)
            offspring = self.generate_offspring(population, offspring_times)
            population = self.select_next_population(population, offspring, self.pop_size)
            self.log()
            if self.config.get('model.experience_prob')>0:
                self.update_experience()
            if len(self.all_mols) >= self.budget or self.early_stopping:
                self.log(finish=True)
                break
            self.num_gen+=1
            data = {
                'history':self.history,
                'init_pops':init_pops,
                'final_pops':population,
                'all_mols':self.mol_buffer,
                'properties':self.property_list,
                'evaluation': self.results_dict['results'],
                'running_time':f'{(time.time()-start_time)/3600:.2f} hours'
            }
            with open(store_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Data saved to {store_path}")
        print(f'=======> total running time { (time.time()-start_time)/3600 :.2f} hours <=======')
        
        return init_pops,population  # 计算效率

    def mating(self,parent_list):
        crossover_prob = self.config.get('model.crossover_prob')
        mutation_prob = self.config.get('model.mutation_prob')
        #crossover_prob = 0.2 + len(self.history_moles) / self.budget * 0.6
        #mutation_prob = 1 - crossover_prob
        explore_prob = self.config.get('model.explore_prob')
        
        
        cycle_length = 10
        #explore_prob = 0.25 + 0.7 * (np.cos(2 * np.pi * self.num_gen / cycle_length) + 1) / 2 
        #crossover_prob = (1 - explore_prob) * (2/3)
        #mutation_prob = 1 - explore_prob - crossover_prob
        
        #print('e pro, cross prob, mutate prob',explore_prob,crossover_prob,mutation_prob)
        function = np.random.choice([self.crossover,self.mutation,self.explore],p=[crossover_prob,
                                                                                   mutation_prob,
                                                                                   explore_prob])
        smiles,prompt,response = function(parent_list)
        return smiles,prompt,response
    
    def generate_offspring(self, population, offspring_times):
        parents = [random.sample(population, 2) for i in range(offspring_times)]
        '''
        parents = []
        scores = np.array([p.total for p in population])
        prob = scores / np.sum(scores)
        for _ in range(offspring_times):
            selected_indices = np.random.choice(len(population), size=2, replace=False, p=prob)
            pair = [population[i] for i in selected_indices]
            parents.append(pair)
        
        population = sorted(population, key=lambda p: p.total, reverse=True)
        n = len(population)
        # 分配权重：rank 1 gets weight n, rank 2 gets n-1, ..., rank n gets weight 1
        weights = np.array([n - i for i in range(n)], dtype=np.float64)
        prob = weights / weights.sum()
        
        parents = []
        for _ in range(offspring_times):
            selected_indices = np.random.choice(n, size=2, replace=False, p=prob)
            parents.append([population[i] for i in selected_indices])
        '''
        parallel = True

        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.mating, parent_list=parent_list) for parent_list in parents]
                results = [future.result() for future in futures]
                children, prompts, responses = zip(*results) #[[item,item],[item,item]] # ['who are you value 1', 'who are you value 2'] # ['yes, 'no']
                self.llm_calls += len(results)
                
        else:
            children,prompts,responses = [],[],[]
            for parent_list in tqdm(parents):
                child,prompt,response = self.mating(parent_list)
                children.append(child)
                prompts.append(prompt)
                responses.append(response)
                self.llm_calls += 1
                    

        offspring = []
        smiles_this_gen = []
        for child_pair in children:
            self.generated_num += len(child_pair)
            for child in child_pair:
                mol = Chem.MolFromSmiles(child.value) 
                if mol is None: # check if valid
                    self.failed_num += 1
                else:
                    child.value = Chem.MolToSmiles(mol)
                    # check if repeated
                    if child.value in self.history_moles or child.value in smiles_this_gen:
                        self.repeat_num +=1
                    else:
                        smiles_this_gen.append(child.value)
                        offspring.append(child)

        if len(offspring) == 0:
            return []
        self.evaluate_all(offspring)
        self.store_history_moles(offspring)
        self.history.push(prompts,children,responses) 
        return offspring

    def select_next_population(self, population, offspring, pop_size):
        combined_population = offspring + population 
        #if len(self.property_list)>1:
        return nsga2_so_selection(combined_population, pop_size)
       
    def no_selection(self,combined_population, pop_size):
        return combined_population[:pop_size] 
    
    def hvc_selection(self,pops,pop_size):
        scores = []
        for pop in pops:
            scores.append(pop.scores)
        scores = np.stack(scores)
        hv_pygmo = pg.hypervolume(scores)
        hvc = hv_pygmo.contributions(np.array([1.0 for i in range(scores.shape[1])]))
        sorted_indices = np.argsort(hvc)[::-1]  # Reverse to sort in descending order
        bestn = [pops[i] for i in sorted_indices[:pop_size]]
        return bestn

    
 