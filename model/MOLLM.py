import yaml
import json
from rewards.system import RewardingSystem
from model.LLM import LLM
import os
import pickle
from algorithm.MOO import MOO
from eval import eval_mo_results,mean_sr
import pandas as pd
class ConfigLoader:
    def __init__(self, config_path="config.yaml"):
        config_path = os.path.join('config',config_path)
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
    
    def to_string(self, config=None, indent=0):
        """Recursively format the configuration dictionary as a string."""
        if config is None:
            config = self.config
        lines = []
        for key, value in config.items():
            if isinstance(value, dict):
                lines.append(" " * indent + f"{key}:")
                lines.append(self.to_string(value, indent + 2))
            else:
                lines.append(" " * indent + f"{key}: {value}")
        return "\n".join(lines)

class MOLLM:
    def __init__(self, config='base.yaml',resume=False,eval=False,seed=42):
        self.config = ConfigLoader(config)
        self.property_list = self.config.get('goals')
        if not eval:
            self.reward_system = RewardingSystem(material=self.config.get('material'))
        self.llm = LLM()
        self.seed = seed
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
        self.yue_smiles = pd.read_csv('/home/v-nianran/src/MOLLM/data/smiles1960.csv').smiles.values
    
    def load_dataset(self):
        with open(self.config.get('dataset.path'), 'r') as json_file:
            self.dataset= json.load(json_file)
    
    def run(self):
        if self.resume:
            self.load_from_pkl(self.save_path)
        for i in range(self.start_index,len(self.dataset['prompts'])):
            print(f'start {i+1}')
            moo = MOO(self.reward_system, self.llm,self.property_list,self.config,self.seed)
            init_pops,final_pops = moo.run(self.dataset['prompts'][i], self.dataset['requirements'][i])
            self.history.append(moo.history)
            self.final_pops.append(final_pops)
            self.init_pops.append(init_pops)
            if (i)%1 ==0:
                self.evaluate() # evaluate self.final_pops and self.init_pops
            self.save_to_pkl(self.save_path)

            self.check_repeat()
    def check_repeat(self):
        time = 0
        for pops in self.final_pops:
            for p in pops:
                if p.value in self.yue_smiles:
                    time += 1
        print(f'There are {time} molecules repeated')


    def load_evaluate(self):
        self.load_from_pkl(self.save_path)
        r = self.evaluate()
        print(r)

    def evaluate(self):
        obj = {
            'init_pops':self.init_pops,
            'final_pops':self.final_pops,
        }
        r  = eval_mo_results(self.dataset,obj,ops=self.property_list)
        mean_success_num,mean_success_rate,new_sr = mean_sr(r)
        print(f'mean success number: {mean_success_num:.4f}, new mean success rate {new_sr:.4f}, mean success rate: {mean_success_rate:.4f}')
        self.results = {
            'mean success num': mean_success_num,
            'new mean success rate': new_sr,
            'mean success rate': mean_success_rate,
            'success num each problem': r
        }
        return r

    def save_to_pkl(self, filepath):
        data = {
            'history':self.history,
            'init_pops':self.init_pops,
            'final_pops':self.final_pops,
            'evaluation':self.results,
            'properties':self.property_list
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
        