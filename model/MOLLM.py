import yaml
import json
from model.LLM import LLM
import os
import pickle
from algorithm.MOO import MOO
from eval import eval_mo_results,mean_sr
import pandas as pd
import importlib
class ConfigLoader:
    def __init__(self, config_path="config.yaml"):
        config_path = os.path.join('problem',config_path)
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
    def __init__(self, config='base.yaml',resume=False,eval=False,seed=42,objectives=None,directions=None):
        if isinstance(config,str):
            self.config = ConfigLoader(config)
        else:
            self.config = config
        print('goals, directs',self.config.get('goals'),self.config.get('optimization_direction'))
        if objectives is not None:
            print(f'objectives  {objectives} directions {directions}')
            self.config.config['goals'] = objectives
            assert directions is not None
            self.config.config['optimization_direction'] = directions
            print('goals, directs',self.config.get('goals'),self.config.get('optimization_direction'))
        self.property_list = self.config.get('goals')
        if not eval:
            module_path = self.config.get('evalutor_path')  # e.g., "molecules"
            module = importlib.import_module(module_path)
            RewardingSystem = getattr(module, "RewardingSystem")
            self.reward_system = RewardingSystem(config=self.config)
        self.llm = LLM(model = self.config.get('model.name'))
        self.seed = seed
        self.history = []
        self.init_pops = []
        self.final_pops = []
        self.start_index = 0
        self.save_dir = self.config.get('save_dir')
        model_name = self.config.get('model.name')
        if ',' in model_name:
            model_name = model_name.split(',')[1]
        self.save_dir = os.path.join(self.save_dir,model_name)
        self.save_suffix = self.config.get('save_suffix')
        self.resume = resume
        self.save_path = os.path.join(self.save_dir,'_'.join(self.property_list) + '_' + self.save_suffix +'.pkl')
        self.results = {
            'mean success num': 0,
            'mean success rate': 0,
            'success num each problem': []
        }
    
    def run(self):
        if self.resume:
            self.load_from_pkl(self.save_path)

        moo = MOO(self.reward_system, self.llm,self.property_list,self.config,self.seed)
        init_pops,final_pops = moo.run()
        self.history.append(moo.history)
        self.final_pops.append(final_pops)
        self.init_pops.append(init_pops)

        #self.evaluate() # evaluate self.final_pops and self.init_pops
        #self.save_to_pkl(self.save_path)

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

    def save_to_pkl(self, filepath,i=0):
        data = {
            'history':self.history,
            'init_pops':self.init_pops,
            'final_pops':self.final_pops,
            'evaluation':self.results,
            'properties':self.property_list
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        if i% 10==0:
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
        