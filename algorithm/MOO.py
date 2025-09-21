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
from algorithm.base import ItemFactory,HistoryBuffer
from openai import AzureOpenAI
from rdkit import Chem
import json
from eval import get_evaluation
import time
from model.util import *
from algorithm import PromptTemplate
import importlib
import pickle
from model.LLM import LLM


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

class MOO:
    def __init__(self, reward_system, llm,property_list,config,seed):
        self.reward_system = reward_system
        self.config = config
        self.seed = seed
        self.llm = llm
        #self.au_llm = LLM(model = self.config.get('model.name2'))
        self.history = HistoryBuffer()
        self.item_factory = ItemFactory(property_list)
        self.property_list = property_list
        self.pop_size = self.config.get('optimization.pop_size')
        self.budget = self.config.get('optimization.eval_budget')
        self.use_au = self.config.get('use_au')
        self.save_dir = os.path.join(self.config.get('save_dir'),self.config.get('model.name'))
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
        self.start_time = time.time()
        self.num_offspring = self.config.get('num_offspring',default=2)

    def generate_initial_population(self, n):
        module_path = self.config.get('evalutor_path')  # e.g., "molecules"
        module = importlib.import_module(module_path)
        _generate = getattr(module, "generate_initial_population")
        strings = _generate(self.config,self.seed)
        if isinstance(strings[0],str):
            return [self.item_factory.create(i) for i in strings]
        else:
            self.store_history_moles(strings)
            return strings

    def mutation(self, parent_list):
        prompt = self.prompt_generator.get_prompt('mutation',parent_list,self.history_moles)
        #print('mutation prompt:',prompt)
        #assert False
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        return [self.item_factory.create(smile) for smile in new_smiles],prompt,response

    def crossover(self, parent_list):
        prompt = self.prompt_generator.get_prompt('crossover',parent_list,self.history_moles)
        #print('crossover prompt:',prompt)
        #assert False
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        
        return [self.item_factory.create(smile) for smile in new_smiles],prompt,response

    def evaluate(self,pops):
        pops, log_dict = self.reward_system.evaluate(pops)
        self.failed_num += log_dict['invalid_num']
        self.repeat_num += log_dict['repeated_num']
        pops = self.store_history_moles(pops)
        return pops

    def store_history_moles(self,pops):
        unique_pop = []
        for i in pops:
            if i.value not in self.history_moles:
                self.history_moles.append(i.value)
                self.mol_buffer.append([i, len(self.mol_buffer)+1])
                unique_pop.append(i)
            else:
                self.repeat_num += 1
        return unique_pop
        
    def explore(self):
        pass 

    def log_results(self, 
                mol_buffer: list = None, 
                buffer_type: str = "default", 
                finish: bool = False) -> None:
        """
        Logs performance metrics (AUCs, top scores, diversity, hypervolume, etc.) 
        for the given molecular buffer, and saves them to JSON.

        Parameters:
        - mol_buffer (list): The molecular buffer to evaluate, list of (Molecule, Info) tuples.
        - buffer_type (str): The name of the buffer type ('main', 'au', or 'default' for mol_buffer).
        - finish (bool): Whether this is the final evaluation.
        """
        if mol_buffer is None:
            mol_buffer = self.mol_buffer
        auc1 = top_auc(mol_buffer, 1, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc10 = top_auc(mol_buffer, 10, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc100 = top_auc(mol_buffer, 100, finish=finish, freq_log=100, max_oracle_calls=self.budget)

        top100 = sorted(mol_buffer, key=lambda item: item[0].total, reverse=True)[:100]
        top100_mols = [i[0] for i in top100]
        top10 = top100_mols[:10]

        avg_top10 = np.mean([i.total for i in top10])
        avg_top100 = np.mean([i.total for i in top100_mols])

        if self.config.get('cal_div',default=False):
            from tdc import Evaluator
            div_evaluator = Evaluator(name = 'Diversity')
            diversity_top100 = div_evaluator([i.value for i in top100_mols])
        else:
            diversity_top100 = 0
        
        ### 
        if 'l_delta_b' in top10[0].property and 'aspect_ratio' in top10[0].property:
            all_mols = [item[0] for item in mol_buffer]
            all_mols = [i for i in all_mols if i.constraints['feasibility']<0.01]
            if len(all_mols)>0:
                scores = np.array([[-i.property['l_delta_b'],i.property['aspect_ratio']] for i in all_mols])
                volume = cal_fusion_hv(scores)
            else:
                volume = 0
        else:
            scores = np.array([i.scores for i in top100_mols])
            volume = cal_hv(scores) ###

        if buffer_type == "default":
            uniqueness = 1 - self.repeat_num / (self.generated_num + 1e-6)
            validity = 1 - self.failed_num / (self.generated_num + 1e-6)
        else:
            uniqueness = 1 - self.record_dict[f'{buffer_type}_repeat_num'] / (self.record_dict[f'{buffer_type}_all_num'] + 1e-6)
            validity = 1 - self.record_dict[f'{buffer_type}_failed_num'] / (self.record_dict[f'{buffer_type}_all_num'] + 1e-6)

        # Handle early stopping for default logging only
        if buffer_type == "default":
            new_score = avg_top100
            if new_score - self.old_score < 1e-4 and self.old_score>0.05:
                self.patience += 1
                if self.config.get('early_stopping',default=True) and self.patience >= 6:
                    print('convergence criteria met, abort ...... ')
                    self.early_stopping = True
            else:
                self.patience = 0
            self.old_score = new_score

        # Select results_dict and path
        if buffer_type == "default":
            results_dict = self.results_dict
            save_dir = os.path.join(self.save_dir, "results")
        elif buffer_type == "main":
            results_dict = self.main_results_dict
            save_dir = os.path.join(self.save_dir, "results_main")
        elif buffer_type == "au":
            results_dict = self.au_results_dict
            save_dir = os.path.join(self.save_dir, "results_au")
        else:
            raise ValueError(f"Unknown buffer_type: {buffer_type}")

        json_path = os.path.join(save_dir, '_'.join(self.property_list) + '_' +
                                self.config.get('save_suffix') + f'_{self.seed}.json')
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        results_dict['results'].append({
            'all_unique_moles': len(self.history_moles),
            'llm_calls': self.llm_calls,
            'Uniqueness': uniqueness,
            'Validity': validity,
            'Training_step': self.time_step,
            'avg_top1': top10[0].total,
            'avg_top10': avg_top10,
            'avg_top100': avg_top100,
            'top1_auc': auc1,
            'top10_auc': auc10,
            'top100_auc': auc100,
            'hypervolume': volume,
            'div': diversity_top100,
            'input_tokens': self.llm.input_tokens,
            'output_tokens': self.llm.output_tokens,
            'generated_num': self.generated_num,
            'running_time[s]': time.time()-self.start_time
        })

        # Only include config and history in default log
        if buffer_type == "default":
            print('================================================================')
            results_dict['params'] = self.config.to_string()
            if len(self.history_experience) > 1:
                results_dict['history_experience'] = [self.history_experience[0], self.history_experience[-1]]

        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f'{buffer_type}: {len(self.history_moles)}/{self.budget} generated: {self.generated_num} | '
            f'mol_buffer: {len(mol_buffer)} | '
            f'Uniqueness: {uniqueness:.4f} | '
            f'Validity: {validity:.4f} | '
            f'llm_calls: {self.llm_calls} | '
            f'Training step: {self.time_step} | '
            f'avg_top1: {top10[0].total:.6f} | '
            f'avg_top10: {avg_top10:.6f} | '
            f'avg_top100: {avg_top100:.6f} | '
            f'top1_auc: {auc1:.4f} | '
            f'top10_auc: {auc10:.4f} | '
            f'top100_auc: {auc100:.4f} | '
            f'hv: {volume:.4f} | '
            f'unique top 100:{len(np.unique([i.value for i in top100_mols]))} | '
            f'input_tokens: {self.llm.input_tokens} | '
            f'output_tokens: {self.llm.output_tokens} | '
            f'running_time: {(time.time()-self.start_time)/3600:.3f} h'
            f'div: {diversity_top100:.4f}'
            )


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

    
    def run(self):
        store_path = os.path.join(self.save_dir,'mols','_'.join(self.property_list) + '_' + self.config.get('save_suffix') + f'_{self.seed}' +'.pkl')
        if not os.path.exists(os.path.dirname(store_path)):
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
            
        ''' initialize genetic gfn model'''
        
        if self.use_au:
            from genetic_gfn.multi_objective.genetic_gfn.run import Genetic_GFN_Optimizer
            from genetic_gfn.multi_objective.run import prepare_optimization_inputs
            args, config_default, oracle = prepare_optimization_inputs()
            self.au_model = Genetic_GFN_Optimizer(args=args)
            self.au_model.setup_model(oracle, config_default)

        """High level logic"""
        print('exper_name',self.config.get('exper_name'))
        set_seed(self.seed)
        start_time = time.time()
        
        #initialization 
        if self.config.get('inject_per_generation'):
            module_path = self.config.get('evalutor_path')  # e.g., "molecules"
            module = importlib.import_module(module_path)
            _get = getattr(module, "get_database")
            database = _get(self.config,n_sample=200)
        if self.config.get('resume'):
            population,init_pops = self.load_ckpt(store_path)
        else:
            population = self.generate_initial_population(n=self.pop_size)
            if population[0].total is None:
                population = self.evaluate(population) # including removing invalid and repeated candidates
            self.log_results()
            init_pops = copy.deepcopy(population)
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
        
        self.prompt_generator = self.prompt_module(self.config)
        
        self.num_gen = 0
        
        while True:
            if self.config.get('inject_per_generation'):
                print('inject!')
                population.extend(random.sample(database,self.config.get('inject_per_generation')))
            offspring_times = max(min(self.pop_size //self.num_offspring, (self.budget -len(self.mol_buffer)) //self.num_offspring),1)
            offspring = self.generate_offspring(population, offspring_times)
            population = self.select_next_population(self.pop_size)
            self.log_results()
            if self.config.get('model.experience_prob')>0 and len(self.mol_buffer)>100:
                self.update_experience()
            if len(self.mol_buffer) >= self.budget or self.early_stopping:
                self.log_results(finish=True)
                if self.use_au:
                    self.log_results(self.main_mol_buffer,buffer_type="main", finish=True)
                    self.log_results(self.au_mol_buffer,buffer_type="au", finish=True)
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

    def record(self, tmp_offspring: list, buffer_type: str) -> list:
        """
        Records valid and unique molecules from the given temporary offspring list.

        Parameters:
        - tmp_offspring (list): List of generated candidate molecules (Item objects).
        - buffer_type (str): Type identifier for tracking stats (e.g., 'main', 'au').

        Returns:
        - offspring (list): Filtered list of valid and unique Item objects.
        """
        offspring = []
        smiles_this_gen = []
        self.record_dict[buffer_type + '_all_num'] += len(tmp_offspring)

        for child in tmp_offspring:
            mol = Chem.MolFromSmiles(child.value)
            if mol is None:
                self.record_dict[buffer_type + '_failed_num'] += 1
            else:
                child.value = Chem.MolToSmiles(mol)
                if (child.value in self.record_dict[buffer_type + '_history_smiles'] or
                    child.value in smiles_this_gen):
                    self.record_dict[buffer_type + '_repeat_num'] += 1
                else:
                    self.record_dict[buffer_type + '_history_smiles'].append(child.value)
                    smiles_this_gen.append(child.value)
                    offspring.append(child)
        return offspring

    def mating(self, parent_list: list, au: bool = False) -> tuple:
        """
        Applies genetic operators (crossover, mutation, or exploration) to parent list.

        Parameters:
        - parent_list (list): A list of parent Item objects.
        - au (bool): Whether this is for auxiliary model (default: False).

        Returns:
        - tuple: (list of offspring Items, str prompt, str response)
        """
        crossover_prob = self.config.get('model.crossover_prob')
        mutation_prob = self.config.get('model.mutation_prob')
        explore_prob = self.config.get('model.explore_prob')
        function = np.random.choice([self.crossover,self.mutation,self.explore],p=[crossover_prob,
                                                                                   mutation_prob,
                                                                                   explore_prob])
        items,prompt,response = function(parent_list)
        return items,prompt,response
    
    def generate_offspring(self, population: list, offspring_times: int) -> list:
        """
        Generates new offspring from the population using LLM-driven operations, and evaluates them.

        Parameters:
        - population (list): List of Item objects representing the current population.
        - offspring_times (int): Number of offspring pairs to generate.

        Returns:
        - list: Evaluated and recorded offspring.
        """
        parents = [random.sample(population, self.num_offspring) for i in range(offspring_times)]
        parallel = True
        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.mating, parent_list=parent_list) for parent_list in parents]
                #results = [future.result() for future in futures]
                #children, prompts, responses = zip(*results) #[[item,item],[item,item]] # ['who are you value 1', 'who are you value 2'] # ['yes, 'no']
                #self.llm_calls += len(results)     
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=900)  # 最多等待 120 秒
                        results.append(result)
                    except concurrent.futures.TimeoutError:
                        print("Warning: A task timed out after 180 seconds.")
                        continue  # 或者 results.append(default_value)
                if results:
                    children, prompts, responses = zip(*results)
                    self.llm_calls += len(results)
                else:
                    print("No results collected due to timeouts.")
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
        
        if self.use_au:
            self.record(tmp_offspring,'main')
            self.save_log_mols(tmp_offspring,buffer_type='main')
            
            au_smiles = self.au_model.sample_n_smiles(32,self.mol_buffer)
            ### au_smiles = self.generate_offspring_au(population,offspring_times=20)
            au_smiles = [self.item_factory.create(smiles) for smiles in au_smiles]
            self.generated_num += len(au_smiles)
            self.record(au_smiles,'au')
            self.save_log_mols(au_smiles,buffer_type='au')
            tmp_offspring.extend(au_smiles)
            
        #offspring = self.sanitize(tmp_offspring,record=True)
        offspring = tmp_offspring
        
        if len(offspring) == 0:
            return []

        offspring = self.evaluate(offspring)
        self.history.push(prompts,children,responses) 
        return offspring

    def save_log_mols(self, mols: list, buffer_type: str) -> None:
        """
        Evaluates molecules, trains AU model if applicable, stores in buffer, and logs metrics.

        Parameters:
        - mols (list): List of Item objects to save and evaluate.
        - buffer_type (str): Indicates which buffer ('main' or 'au') to update.
        """
        self.time_step += 1
        mols,_ = self.reward_system.evaluate(mols)
        if buffer_type=='main':
            mol_buffer = self.main_mol_buffer
            self.au_model.train_on_smiles([i.value for i in mols],[i.total for i in mols],loop=4,time_step=self.time_step,mol_buffer=mol_buffer)
        elif buffer_type=='au':
            mol_buffer = self.au_mol_buffer
            self.au_model.train_on_smiles([i.value for i in mols],[i.total for i in mols],loop=4,time_step=self.time_step,mol_buffer=mol_buffer)
        # print('oracle length: ',len(self.mol_buffer))

        self.mol_buffer_store(mol_buffer,mols)
        # 这里加了重复的 
        self.log_results(mol_buffer,buffer_type, finish=False)

    def mol_buffer_store(self, mol_buffer: list, mols: list) -> list:
        """
        Stores non-duplicate molecules into a specified buffer.

        Parameters:
        - mol_buffer (list): List buffer to store Item objects.
        - mols (list): List of Item objects to attempt storing.

        Returns:
        - list: Updated mol_buffer with new unique molecules.
        """
        all_smiles = [i[0].value for i in mol_buffer]
        for child in mols:
            mol = Chem.MolFromSmiles(child.value) 
            if mol is None: # check if valid
                pass
            else:
                child.value = Chem.MolToSmiles(mol,canonical=True)
                # check if repeated
                if child.value in all_smiles:
                    pass
                else:
                    all_smiles.append(child.value)
                    mol_buffer.append([child,len(mol_buffer)+1])
        return mol_buffer


    def select_next_population(self,pop_size):
        whole_population = [i[0] for i in self.mol_buffer]
        if len(self.property_list)>1:
            return nsga2_so_selection(whole_population, pop_size)
        else:
            return so_selection(whole_population,pop_size)

    def generate_offspring_au(self, population, offspring_times=20):
        parents = [random.sample(population, 2) for i in range(offspring_times)]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.mating, parent_list=parent_list,au=True) for parent_list in parents]
            results = [future.result() for future in futures]
            children, prompts, responses = zip(*results) #[[item,item],[item,item]] # ['who are you value 1', 'who are you value 2'] # ['yes, 'no']
            self.llm_calls += len(results)
        tmp_offspring = []
        for child_pair in children:
            self.generated_num += len(child_pair)
            tmp_offspring.extend(child_pair)
        return tmp_offspring
    
    def load_ckpt(self,store_path):
        print('resume training')
        save_dir = os.path.join(self.save_dir, "results")
        json_path = os.path.join(save_dir, '_'.join(self.property_list) + '_' +
                            self.config.get('save_suffix') + f'_{self.seed}.json')
        with open(store_path,'rb') as f:
            ckpt = pickle.load(f)
        with open(json_path,'rb') as f:
            result_ckpt = json.load(f)
        self.mol_buffer = ckpt['all_mols']
        population = self.select_next_population(self.pop_size) 
        init_pops = ckpt['init_pops']
        self.history = ckpt['history']
        
        self.history_moles = [i[0].value for i in self.mol_buffer]
        self.results_dict['results'] = ckpt['evaluation']
        self.generated_num = result_ckpt['results'][-1]['generated_num']
        self.llm_calls = result_ckpt['results'][-1]['llm_calls']
        self.repeat_num = int((1-result_ckpt['results'][-1]['Uniqueness']) * self.generated_num)
        self.failed_num = int((1-result_ckpt['results'][-1]['Validity']) * self.generated_num)
        return population, init_pops

    
    
 