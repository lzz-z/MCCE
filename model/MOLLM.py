import yaml
import json
from rewards.system import Rewarding_system
from model.LLM import LLM
import os
import pickle
from algorithm.MOO import MOO
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
    def __init__(self, config='config/base.yaml'):
        self.config = ConfigLoader(config)
        self.property_list = self.config.get('goals')
        self.reward_system = Rewarding_system()
        self.llm = LLM()
        self.history = []
        self.load_dataset()
        self.init_pops = []
        self.final_pops = []
        self.save_dir = self.config.get('save_dir')
        self.save_suffix = self.config.get('save_suffix')
    
    def load_dataset(self):
        with open(self.config.get('dataset.path'), 'r') as json_file:
            self.dataset= json.load(json_file)
    
    def run(self):
        for i in range(len(self.dataset['prompts'])):
            moo = MOO(self.reward_system, self.llm,self.property_list,self.config)
            init_pops,final_pops = moo.run(self.dataset['prompts'][i], self.dataset['requirements'][i])
            self.history.append(moo.history)
            self.final_pops.append(final_pops)
            self.init_pops.append(init_pops)
            self.save_to_pkl(os.path.join(self.save_dir,'_'.join(self.property_list) + self.save_suffix +'.pkl'))

    def save_to_pkl(self, filepath):
        data = {
            'history':self.history,
            'init_pops':self.init_pops,
            'final_pops':self.final_pops
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filepath}")

    def load_from_pkl(self,filepath):
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f"Data loaded from {filepath}")
        return obj
        