import yaml
import json
from rewards.system import Rewarding_system
from model.LLM import LLM
import os
import pickle
from algorithm.MOO import MOO
from eval import eval_mo_results,mean_sr
class ConfigLoader:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        if value == {}:
            return default
        return value

class MOLLM:
    def __init__(self, config='config/base.yaml',resume=False):
        self.config = ConfigLoader(config)
        self.property_list = self.config.get('goals')
        self.reward_system = Rewarding_system()
        self.llm = LLM()
        self.history = []
        self.load_dataset()
        self.init_pops = []
        self.final_pops = []
        self.start_index = 0
        self.save_dir = self.config.get('save_dir')
        self.save_suffix = self.config.get('save_suffix')
        self.resume = resume
        self.save_path = os.path.join(self.save_dir,'_'.join(self.property_list) + '_' + self.save_suffix +'.pkl')
        self.results = {
            'mean success num': 0,
            'mean success rate': 0,
            'success num each problem': []
        }
    
    def load_dataset(self):
        with open(self.config.get('dataset.path'), 'r') as json_file:
            self.dataset= json.load(json_file)
    
    def run(self):
        if self.resume:
            self.load_from_pkl(self.save_path)
        for i in range(self.start_index,len(self.dataset['prompts'])):
            print(f'start {i+1}')
            moo = MOO(self.reward_system, self.llm,self.property_list,self.config)
            init_pops,final_pops = moo.run(self.dataset['prompts'][i], self.dataset['requirements'][i])
            self.history.append(moo.history)
            self.final_pops.append(final_pops)
            self.init_pops.append(init_pops)
            self.evaluate() # evaluate self.final_pops and self.init_pops
            self.save_to_pkl(self.save_path)
            
    
    def evaluate(self):
        obj = {
            'init_pops':self.init_pops,
            'final_pops':self.final_pops,
        }
        r  = eval_mo_results(self.dataset,obj,similarity_requ=0.4,ops=self.property_list,candidate_num=20)
        mean_success_num,mean_success_rate,new_sr = mean_sr(r)
        print(f'mean success number: {mean_success_num:.4f}, new mean success rate {new_sr:.4f}, mean success rate: {mean_success_rate:.4f}')
        self.results = {
            'mean success num': mean_success_num,
            'new mean success rate': new_sr,
            'mean success rate': mean_success_rate,
            'success num each problem': r
        }

    def save_to_pkl(self, filepath):
        data = {
            'history':self.history,
            'init_pops':self.init_pops,
            'final_pops':self.final_pops,
            'evaluation':self.results
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filepath}")

    def load_from_pkl(self,filepath):
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        self.history = obj['history']
        self.init_pops = obj['init_pops']
        self.final_pops = obj['final_pops']
        self.start_index = len(obj['init_pops'])
        print(f"Data loaded from {filepath}")
        return obj
        