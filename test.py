'''
from rewards.system import RewardingSystem
s = RewardingSystem()
from eval import judge
mols = [['CCH','CCOCCOCC'],['CCOCCOCC','C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']]
ops = ['sa','reduction_potential','smarts_filter','logs']

#print(s.evaluate(ops,mols))
from tdc.generation import MolGen
import random
import numpy as np
import json
seed = 42
random.seed(seed)
np.random.seed(seed)
data = MolGen(name='ZINC')
split = data.get_split()
moles_df = split['train']

smiles = moles_df.sample(100000).smiles.values
bad_smiles = []
index = 0
while True:
    this_smiles = smiles[index:index+10]
    index+=10
    smiles_list = [[smiles[0],i] for i in this_smiles]
    results = s.evaluate(ops,smiles_list)
    for i in range(len(this_smiles)):
        if results['sa'][i] >3 and results['reduction_potential'][i] < -2.3 \
            and results['smarts_filter'][i] == 1 and results['logs'][i]<-2:
            bad_smiles.append(this_smiles[i])
    print('len:',len(bad_smiles))
    if len(bad_smiles)>10:
        with open("bad_smiles_chem.json", "w") as f:
            json.dump(bad_smiles, f, indent=4)
    if len(bad_smiles) >= 300:
        bad_smiles = bad_smiles[:300]
        break
with open("bad_smiles_chem.json", "w") as f:
    json.dump(bad_smiles, f, indent=4)
'''
'''
sa 1.5960526085782512 6.649504345814619   min
reduction_potential -4.990693092346191 -0.9798483848571777    to -1.3
smarts_filter 0.0 1.0    # 0 pass
logs -7.43935489654541 0.47240304946899414 # max
'''
'''
import pandas as pd
from algorithm.MOO import MOO
from algorithm.base import Item
from model.MOLLM import ConfigLoader
from rewards.system import RewardingSystem
#smiles_df = pd.read_csv('/home/v-nianran/src/MOLLM/data/smiles1960.csv')
#smiles = smiles_df.smiles.values
from tdc.generation import MolGen
data = MolGen(name='ZINC')
split = data.get_data()
smiles = split.smiles.values

config = ConfigLoader('/home/v-nianran/src/MOLLM/config/yue/sa_drd2_qed_2.yaml')
property_list = config.get('goals')
print(property_list)
moo = MOO(RewardingSystem(use_tqdm=True,chunk_size=1000), llm=None,property_list=property_list,config=config,seed=42)
items = [Item(i,property_list) for i in smiles]
moo.original_mol = items[0]
requirement = {
            "qed_requ": {
                "property": "QED",
                "requirement": "increase"
            },
            "drd2_requ": {
                "property": "DRD2",
                "requirement": "decrease"
            },
            "sa_requ": {
                "property": "SA",
                "requirement": "decrease"
            },
            "gsk3b_requ": {
                "property": "GSK3\u03b2",
                "requirement": "decrease"
            },
            "jnk3_requ": {
                "property": "JNK3",
                "requirement": "increase"
            }
        }
moo.requirement_meta = requirement
moo.evaluate_all(items)
#best300 = moo.select_next_population(items,[],300)
filepath = 'data/zinc250_5goals.pkl'
#for mol in items:
#    print(mol.property)
import pickle
def save_to_pkl( filepath):
    data = {
        'all':items
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filepath}")
save_to_pkl(filepath)
'''
'''
import pandas as pd
from algorithm.MOO import MOO
from algorithm.base import Item
from model.MOLLM import ConfigLoader
from rewards.system import RewardingSystem
from eval import extract_smiles_from_string

def eval_strings(input_path,output_path):
    with open(input_path,'r') as f:
        data = f.readlines()
    smiles = []
    for line in data:
        if '<mol>' in line:
            smiles.extend(extract_smiles_from_string(line))
    smiles

    #smiles_df = pd.read_csv('/home/v-nianran/src/MOLLM/data/smiles1960.csv')
    #smiles = smiles_df.smiles.values
    config = ConfigLoader('/home/v-nianran/src/MOLLM/config/chem/filter_logs_red_sa.yaml')
    property_list = config.get('goals')
    print(property_list)
    moo = MOO(RewardingSystem(), llm=None,property_list=property_list,config=config)
    items = [Item(i,property_list) for i in smiles]
    moo.original_mol = items[0]
    import json
    with open("/home/v-nianran/src/MOLLM/data/yue300.json", 'r') as json_file:
        dataset= json.load(json_file)
    requirement = dataset['requirements'][0]
    moo.requirement_meta = requirement
    moo.evaluate_all(items)
    filepath = output_path
    import pickle
    def save_to_pkl( filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(items, f)
        print(f"Data saved to {filepath}")
    save_to_pkl(filepath)

import os
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, get_bearer_token_provider
class LLM:
    def __init__(self,model='chatgpt'):
        self.model_choice = model
        self.model = self._init_model(model)
        self.chat = self._init_chat(model)

    def _init_chat(self,model):
        if model == 'chatgpt':
            return self.gpt_chat

    def _init_model(self,model):
        if model == 'chatgpt':
            return self._init_chatgpt()

    def _init_chatgpt(self):
        # Set the necessary variables
        resource_name = "gcrgpt4aoai2c"
        endpoint = f"https://{resource_name}.openai.azure.com/"
        api_version = "2024-02-15-preview"  # Replace with the appropriate API version

        azure_credential = ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential=True,
                # Exclude other credentials we are not interested in.
                exclude_environment_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_developer_cli_credential=True,
                exclude_powershell_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credentials=True,
                managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
            )
        )

        token_provider = get_bearer_token_provider(azure_credential,
            "https://cognitiveservices.azure.com/.default")
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider
        )
        
        
        return client
    
    def gpt_chat(self, content):
        message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]
        message.append({"role":"user", "content":content})
        completion = self.model.chat.completions.create(
            model="gpt-4o",
            max_tokens = 2048,
            temperature=0,
            messages=message,
        )
        res = completion.choices[0].message.content
        return res

if __name__ == '__main__':
    #input_path = '/home/v-nianran/src/MOLLM/o1_answers.txt'
    #output_path = 'data/o1_300.pkl'
    #eval_strings(input_path,output_path)
    llm = LLM()
    print(llm.chat('who are you?'))
'''

from algorithm.PromptTemplate import Prompt
import pickle
with open('/home/v-nianran/src/results/sa_drd2_qed_gsk3b_jnk3_best100_mutation_0.5_experience_bestworst_46.pkl','rb') as f:
    obj = pickle.load(f)
pops = obj['final_pops']
properties = ['sa','drd2','qed','gsk3b','jnk3']
requirement = {
            "qed_requ": {
                "property": "QED",
                "requirement": "increase"
            },
            "drd2_requ": {
                "property": "DRD2",
                "requirement": "decrease"
            },
            "sa_requ": {
                "property": "SA",
                "requirement": "decrease"
            },
            "gsk3b_requ": {
                "property": "GSK3\u03b2",
                "requirement": "decrease"
            },
            "jnk3_requ": {
                "property": "JNK3",
                "requirement": "increase"
            }
        }
original_mol = pops[0]
pg = Prompt(original_mol,requirement,properties)
print(pg.get_exploration_prompt(pops))