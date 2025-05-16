import numpy as np
class Item:
    #property_list = ['qed', 'logp', 'donor']

    def __init__(self, value, property_list):
        self.value = value
        self.property_list = property_list if property_list is not None else self.property_list
        # raw scores are the original objective values
        self.assign_raw_scores([ 0 for prop in self.property_list])
        # scores are the objective values (after judgement) for MOO
        self.scores = [ 0 for prop in self.property_list]
    
    def assign_raw_scores(self,scores):
        self.raw_scores = scores
        self.property = {self.property_list[i]:scores[i] for i in range(len(self.property_list))}
        self.cal_sum()
    
    def cal_sum(self):
        self.total = 0
        for p in self.property_list:
            if p in ['qed','jnk3','drd2','bbbp1','gsk3b','celecoxib_rediscovery', 'troglitazone_rediscovery','thiothixene_rediscovery',
         'albuterol_similarity','mestranol_similarity',
        'isomers_c7h8n2o2','isomers_c9h10n2o2pf2cl','median1','median2', 'osimertinib_mpo',
        'fexofenadine_mpo','ranolazine_mpo','perindopril_mpo', 'amlodipine_mpo',
        'sitagliptin_mpo','zaleplon_mpo','valsartan_smarts', 'deco_hop', 'scaffold_hop']:
                self.total += self.property[p]
            elif p == 'sa':
                self.total += 1- (self.property[p] -1 ) /9
            elif p in ['gsk3b','drd2','smarts_filter']:
                self.total += 1-self.property[p]
            elif p in ['logs']:
                self.total += (self.property[p] + 8 ) / (9)
            elif p in ['reduction_potential']:
                self.total += 1-  abs( np.clip(self.property[p],-2.3,-0.3) +1.3)
            else:
                raise NotImplementedError(f"{p} property is not defined in base.py")
                
import pickle
import os
class HistoryBuffer:
    def __init__(self):
        self.prompts = [] #
        self.generations = [] # 
        self.responses = [] # <mol> </mol>
        self.save_path = 'checkpoint/'
        self.successful_molecules = []
        self.failed_molecules = []

    # 

    def save_to_pkl(self, filename):
        with open(os.path.join(self.save_path,filename), 'wb') as f:
            pickle.dump(self, f)
        print(f"Data saved to {filename}")

    def load_from_pkl(self,filename):
        with open(os.path.join(self.save_path,filename), 'rb') as f:
            obj = pickle.load(f)
        print(f"Data loaded from {filename}")
        return obj

    def push(self,prompts,generation,responses):
        self.prompts.append(prompts)
        self.generations.append(generation)
        self.responses.append(responses)