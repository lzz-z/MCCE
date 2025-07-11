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
from model.util import nsga2_so_selection,top_auc,cal_hv
from algorithm import PromptTemplate
from eval import judge
import pygmo as pg
import pickle

from genetic_gfn.multi_objective.genetic_gfn.run import Genetic_GFN_Optimizer
from genetic_gfn.multi_objective.run import prepare_optimization_inputs

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
        self.use_au = self.config.get('use_au')
        self.save_dir = os.path.join(self.config.get('save_dir'),self.config.get('model.name'))
        self.init_mol_dataset()
        self.prompt_module = getattr(PromptTemplate ,self.config.get('model.prompt_module',default='Prompt'))
        self.history_moles = []
        self.mol_buffer = [] # same as all_mols but with orders for computing auc
        self.main_mol_buffer = []
        self.au_mol_buffer = []
        self.results_dict = {'results':[]}
        self.main_results_dict = {'results':[]}
        self.au_results_dict = {'results':[]}
        self.history_experience = []
        self.repeat_num = 0
        self.failed_num = 0
        self.generated_num = 0
        self.llm_calls = 0
        self.patience = 0
        self.old_score = 0
        self.early_stopping = False
        self.record_dict = {}
        for i in ['main','au']:
            for j in ['all_num','failed_num','repeat_num']:
                self.record_dict[i+'_'+j] = 0
        self.record_dict['main_history_smiles'] = []
        self.record_dict['au_history_smiles'] = []
        self.time_step = 0

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
      
        response = self.llm.chat(prompt)
        #print('response:',response,'\n\n\n')
        #assert False
        
        new_smiles = extract_smiles_from_string(response)
        return [Item(smile,self.property_list) for smile in new_smiles],prompt,response
    
    def explore(self, parent_list):
        # Deprecated
        prompt = self.prompt_generator.get_prompt('explore',parent_list,self.mol_buffer)
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
            else:
                print('this place should not have repeated smiles')
                assert False

    def log(self,finish=False):
        auc1 = top_auc(self.mol_buffer, 1, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc10 = top_auc(self.mol_buffer, 10, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc100 = top_auc(self.mol_buffer, 100, finish=finish, freq_log=100, max_oracle_calls=self.budget)

        top100 = sorted(self.mol_buffer, key=lambda item: item[0].total, reverse=True)[:100]
        top100 = [i[0] for i in top100]
        top10 = top100[:10]
        avg_top10 = np.mean([i.total for i in top10])
        avg_top100 = np.mean([i.total for i in top100])
        
        diversity_top100 = self.reward_system.all_evaluators['diversity']([i.value for i in top100])

        scores = np.array([i.scores for i in top100])
        volume = cal_hv(scores)

        new_score = avg_top100
        # import ipdb; ipdb.set_trace()
        if new_score == self.old_score:
            self.patience += 1
            if self.patience >= 6:
                print('convergence criteria met, abort ...... ')
                self.early_stopping = True
        else:
            self.patience = 0
        self.old_score = new_score
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
            f'training step:{self.time_step} |'
            f'all generated: {self.generated_num}|' 
            f'avg_top1: {top10[0].total:.4f} | '
            f'avg_top10: {avg_top10:.4f} | '
            f'avg_top100: {avg_top100:.4f} | '
            f'top1_auc : {auc1:.4f} | '
            f'top10_auc : {auc10:.4f} | '
            f'top100_auc : {auc100:.4f} | '
            f'hv: {volume:.4f} | '
            f'div: {diversity_top100:.4f}')
        print('================================================================')
        
        json_path = os.path.join(self.save_dir,"results",'_'.join(self.property_list) + '_' + self.config.get('save_suffix') + f'_{self.seed}'+'.json')
        if not os.path.exists(os.path.dirname(json_path)):
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            
        self.results_dict['params'] = self.config.to_string()
        if len(self.history_experience)> 1:
            # only store the first and the last experience
            self.results_dict['history_experience']  = [self.history_experience[0]] + [self.history_experience[-1]]
        with open(json_path,'w') as f:
            json.dump(self.results_dict, f, indent=4)
        
    def log_mol_buffer(self,mol_buffer,buffer_type, finish=False):
        auc1 = top_auc(mol_buffer, 1, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc10 = top_auc(mol_buffer, 10, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc100 = top_auc(mol_buffer, 100, finish=finish, freq_log=100, max_oracle_calls=self.budget)

        top100 = sorted(mol_buffer, key=lambda item: item[0].total, reverse=True)[:100]
        top100 = [i[0] for i in top100]
        top10 = top100[:10]
        avg_top10 = np.mean([i.total for i in top10])
        avg_top100 = np.mean([i.total for i in top100])
        
        diversity_top100 = self.reward_system.all_evaluators['diversity']([i.value for i in top100])

        scores = np.array([i.scores for i in top100])
        volume = cal_hv(scores)
        uniqueness = 1- self.record_dict[buffer_type+'_repeat_num'] / self.record_dict[buffer_type+'_all_num']
        validity = 1- self.record_dict[buffer_type+'_failed_num'] / self.record_dict[buffer_type+'_all_num']
        
        print(f'{buffer_type}: all num{self.record_dict[buffer_type+"_all_num"]} | '
            f'mol_buffer length: {len(mol_buffer)}  | '
            f'Uniqueness:{uniqueness:.4f} | '
            f'Validity:{validity:.4f} | '
            f'Training step:{self.time_step}  |'
            f'all generated:{self.generated_num}  |'             #f'Novelty:{1-already/(self.generated_num+1e-6):.4f} | '
            f'avg_top1: {top10[0].total:.4f} | '
            f'avg_top10: {avg_top10:.4f} | '
            f'avg_top100: {avg_top100:.4f} | '
            f'top1_auc : {auc1:.4f} | '
            f'top10_auc : {auc10:.4f} | '
            f'top100_auc : {auc100:.4f} | '
            f'hv: {volume:.4f} | '
            f'div: {diversity_top100:.4f}')
        
        json_path = os.path.join(self.save_dir,"results_"+buffer_type,'_'.join(self.property_list) + '_' + self.config.get('save_suffix') + f'_{self.seed}'+'.json')
        if not os.path.exists(os.path.dirname(json_path)):
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        if buffer_type=='main':
            results_dict = self.main_results_dict
        elif buffer_type=='au':
            results_dict = self.au_results_dict
        results_dict['results'].append(
            {   'all_unique_moles': len(self.history_moles),
                'llm_calls': self.llm_calls,
                'Uniqueness':uniqueness,
                'Validity':validity,
                'Training_step':self.time_step,
                'all_generated': self.generated_num,
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
        with open(json_path,'w') as f:
            json.dump(results_dict, f, indent=4)        

    def update_experience(self):
        prompt,best_moles_prompt,bad_moles_prompt = self.prompt_generator.make_experience_prompt(self.mol_buffer)
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
        ''' initialize genetic gfn model'''
        args, config_default, oracle = prepare_optimization_inputs()
        self.au_model = Genetic_GFN_Optimizer(args=args)
        self.au_model.setup_model(oracle, config_default)

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
            offspring_times = max(min(self.pop_size //2, (self.budget -len(self.mol_buffer)) //2),1)
            offspring = self.generate_offspring(population, offspring_times)
            population = self.select_next_population(population, offspring, self.pop_size)
            self.log()
            if self.config.get('model.experience_prob')>0:
                self.update_experience()
            if len(self.mol_buffer) >= self.budget or self.early_stopping:
                self.log(finish=True)
                if self.use_au:
                    self.log_mol_buffer(self.main_mol_buffer,buffer_type="main", finish=True)
                    self.log_mol_buffer(self.au_mol_buffer,buffer_type="au", finish=True)
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
            if self.num_gen%10==0:
                print(f"Data saved to {store_path}")
        print(f'=======> total running time { (time.time()-start_time)/3600 :.2f} hours <=======')
        
        return init_pops,population  # 计算效率

    def sanitize(self,tmp_offspring,record=True):
        offspring = []
        smiles_this_gen = []
        for child in tmp_offspring:
            mol = Chem.MolFromSmiles(child.value) 
            if mol is None: # check if valid
                if record:
                    self.failed_num += 1
            else:
                child.value = Chem.MolToSmiles(mol,canonical=True)
                # check if repeated
                if child.value in self.history_moles or child.value in smiles_this_gen:
                    if record:
                        self.repeat_num +=1
                    else:
                        offspring.append(child)
                else:
                    smiles_this_gen.append(child.value)
                    offspring.append(child)
        return offspring

    def record(self,tmp_offspring,buffer_type):
        offspring = []
        smiles_this_gen = []
        self.record_dict[buffer_type+'_all_num'] += len(tmp_offspring)
        for child in tmp_offspring:
            mol = Chem.MolFromSmiles(child.value) 
            if mol is None: # check if valid
                self.record_dict[buffer_type+'_failed_num'] += 1
            else:
                child.value = Chem.MolToSmiles(mol)
                # check if repeated
                if child.value in self.record_dict[buffer_type+'_history_smiles'] or child.value in smiles_this_gen:

                    self.record_dict[buffer_type+'_repeat_num']+=1
                else:
                    self.record_dict[buffer_type+'_history_smiles'].append(child.value)
                    smiles_this_gen.append(child.value)
                    offspring.append(child)
        return offspring

    def mating(self,parent_list):
        crossover_prob = self.config.get('model.crossover_prob')
        mutation_prob = self.config.get('model.mutation_prob')
        #crossover_prob = 0.2 + len(self.history_moles) / self.budget * 0.6
        #mutation_prob = 1 - crossover_prob
        explore_prob = self.config.get('model.explore_prob')
        
        
        #cycle_length = 10
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
        parallel = True
        '''
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
        
        tmp_offspring = []
        for child_pair in children:
            self.generated_num += len(child_pair)
            tmp_offspring.extend(child_pair)
        '''
        ##########
        prompts = []
        responses = []
        children = []
        if self.use_au:
            if len(self.mol_buffer) > 0:
                au_smiles = self.au_model.sample_n_smiles(32,self.mol_buffer)
                tmp_offspring = [Item(smiles,self.property_list) for smiles in au_smiles]
                self.generated_num += len(au_smiles)
        ############

        if self.use_au:
            self.record(tmp_offspring,'main')
            self.save_log_mols(tmp_offspring,buffer_type='main')
            if len(self.mol_buffer) > 0:
                au_smiles = self.au_model.sample_n_smiles(32,self.mol_buffer)
                au_smiles = [Item(smiles,self.property_list) for smiles in au_smiles]
                self.generated_num += len(au_smiles)
                self.record(au_smiles,'au')
                self.save_log_mols(au_smiles,buffer_type='au')
                tmp_offspring.extend(au_smiles)

        offspring = self.sanitize(tmp_offspring,record=True)
        
        au_smiles = [i[0] for i in self.au_model.experience.memory]
        repeat_au = len(au_smiles) - len(np.unique(au_smiles))
        print('repeat in au:', repeat_au)
        if len(offspring) == 0:
            return []

        self.evaluate_all(offspring)
        #if self.use_au:
        #    self.au_model.train_on_smiles([i.value for i in offspring],[i.total for i in offspring])
        self.store_history_moles(offspring)
        self.history.push(prompts,children,responses) 
        return offspring
    
    def save_log_mols(self,mols,buffer_type):
        self.time_step += 1
        mols = self.sanitize(mols,record=False)
        self.evaluate_all(mols)
        if buffer_type=='main':
            mol_buffer = self.main_mol_buffer
            self.au_model.train_on_smiles([i.value for i in mols],[i.total for i in mols],loop=4,time_step=self.time_step)
        elif buffer_type=='au':
            mol_buffer = self.au_mol_buffer
            self.au_model.train_on_smiles([i.value for i in mols],[i.total for i in mols],loop=4,time_step=self.time_step)
        for mol in mols:
            mol_buffer.append([mol, len(mol_buffer)+1])
        
        self.log_mol_buffer(mol_buffer,buffer_type, finish=False)



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
        hvc = hv_pygmo.contributions(np.array([1.1 for i in range(scores.shape[1])]))
        sorted_indices = np.argsort(hvc)[::-1]  # Reverse to sort in descending order
        bestn = [pops[i] for i in sorted_indices[:pop_size]]
        return bestn

    
 