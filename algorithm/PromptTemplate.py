from model.util import nsga2_selection
import pygmo as pg
import numpy as np
class Prompt:
    def __init__(self,original_mol,requirements,properties,experience_prob=0.5):
        self.requirements = requirements
        self.property = properties
        self.original_mol = original_mol
        self.experience = None
        self.pure_experience = None
        self.exp_times = 0
        self.experience_prob = experience_prob

    def get_first_prompt(self):
        pass

    def get_final_prompt(self):
        pass 
    # 
    def get_prompt(self,prompt_type,ind_list,history_moles):
        if np.random.random() < self.experience_prob and self.experience is not None:
            experience = self.experience
        else:
            experience = ""
        if prompt_type=='crossover':
            prompt = self.get_crossover_prompt(ind_list,history_moles,experience=experience)
        elif prompt_type == 'mutation':
            prompt = self.get_mutation_prompt(ind_list,history_moles,experience=experience)
        elif prompt_type == 'explore':
            prompt = self.get_exploration_prompt(ind_list,history_moles)
        else:
            raise NotImplementedError('not implemented type of operation:',prompt_type)
        
        
        return prompt
            
    def get_crossover_prompt(self,ind_list,history_moles,experience):
        requirement_prompt = self.make_requirement_prompt(self.original_mol,self.requirements,self.property)
        
        history_prompt = self.make_history_prompt(ind_list)
        instruction_prompt = self.make_instruction_prompt(oper_type='crossover')
        description_prompt = self.make_description_prompt()
        final_prompt = requirement_prompt + description_prompt + experience + history_prompt +  instruction_prompt 

        #return "give me 3 molecules with high QED,Do not write code. Do not give any explanation.\
        #      Each output new molecule must start with <mol> and end with </mol> in SIMLES form."
        return final_prompt

    def get_mutation_prompt(self,ind_list,history_moles,experience):
        requirement_prompt = self.make_requirement_prompt(self.original_mol,self.requirements,self.property)
        
        history_prompt = self.make_history_prompt(ind_list[:1])
        instruction_prompt = self.make_instruction_prompt(oper_type='mutation') 
        description_prompt = self.make_description_prompt()
        final_prompt = requirement_prompt + description_prompt + experience + history_prompt + instruction_prompt 
        return final_prompt
    
    def get_exploration_prompt(self,ind_list,history_moles):
        top100 = sorted(history_moles, key=lambda item: item.total, reverse=True)[:100]
        worst10 = sorted(history_moles, key=lambda item: item.total, reverse=True)[-(self.exp_times+1)*10:-(self.exp_times)*10]
        requirement_prompt = self.make_requirement_prompt(self.original_mol,self.requirements,self.property)
        random10 = np.random.choice(top100,size=10,replace=False)
        history_prompt = self.make_history_prompt(random10)
        bad_history_prompt = self.make_history_prompt(worst10,experience=False)
        description_prompt = self.make_description_prompt()
        instruction_prompt = self.make_instruction_prompt(oper_type='explore')
        #final_prompt = requirement_prompt + description_prompt +  history_prompt +  instruction_prompt 
        final_prompt = requirement_prompt + description_prompt + instruction_prompt 
        final_prompt = final_prompt + "There are also some bad molecules, don't propose molecules like the molecules below: \n"
        final_prompt = final_prompt + bad_history_prompt
        
        return final_prompt


    

    def make_description_prompt(self):
        prompt = ""
        for p in self.property:
            prompt += p + ': ' + descriptions[p] + '\n'
        return prompt

    def make_history_prompt(self, ind_list, experience=False): # parent
        if experience:
            pop_content = ""
        else:
            pop_content = "I have some molecules with their objective values. The total score is the integrate of all property values, a higher total score means better molecule. \n"
        for ind in ind_list:
            pop_content += f"<mol>{ind.value}</mol>, its property values are: "
            for index,property in enumerate(ind.property_list):
                pop_content += f"{property}:{ind.raw_scores[index]:.4f},  "
            pop_content += f'total: {ind.total:.4f}\n'
        return pop_content
    
    def make_instruction_prompt(self,oper_type='crossover'): # improvement score = point hypervolume 
        # oper_type : ['crossover', 'mutation', 'explore']
        if oper_type=='mutation':
            prompt = ("Generate 2 new better molecules in SMILES format through mutation, ensuring they are different from all points provided "
                      "above and are not dominated by any of the above.  \n"
            "The molecules must be valid. There are some example operations: \n"
            "1. Modify functional groups selectively while preserving the overall structure. \n"
            "2. Replace atoms or bonds (e.g., replace a hydrogen with a halogen or adjust bond orders) to improve specific properties. \n"
            "3. Add or remove small substituents (e.g., methyl, hydroxyl groups) to explore variations. \n"
            "4. Introduce ring modifications (e.g., add, remove, or modify aromatic or aliphatic rings) to affect stability or reactivity. \n"
            "5. Alter stereochemistry or isomer configurations to explore stereoisomer advantages. \n"
            "6. Consider property-specific optimizations (e.g., hydrophobicity, solubility, binding affinity) to maintain balance. \n"
            "Do not write code. Do not give any explanation. Each output new molecule must start with <mol> and end with </mol> in SIMLES form. \n"
            )
        elif oper_type=='crossover':
            prompt = ("Give me 2 new better molecules that are different from all points above, and not dominated by any of the above. \n"
            "You can do it by applying crossover on the given points and based on your knowledge. The molecule should be valid. \n"
            "Do not write code. Do not give any explanation. Each output new molecule must start with <mol> and end with </mol> in SIMLES form. \n"
            )
        elif oper_type=='explore':
            prompt = ("Confidently propose two novel and better molecules different from the given ones, leveraging your expertise, "
                     #"try not be dominated by any of the above. \n"
                     "The molecule should be valid. \n"
                     "Do not write code. Do not give any explanation. Each output new molecule must start with <mol> and end with </mol> in SIMLES form.\n "
                     )
        else:
            raise NotImplementedError(f'unsupported instruction type: {oper_type}')
        return prompt
    
    def make_experience_prompt(self,all_mols):
        experience_type = np.random.choice(['best_f','hvc','pareto'],p=[0.5,0,0.5])
        #print('expereince_type',experience_type,'wrost index',-(self.exp_times+1)*10,-(self.exp_times)*10)
        worst10 = sorted(all_mols, key=lambda item: item.total, reverse=True)[-(self.exp_times+1)*10:-(self.exp_times)*10]
        if experience_type == 'best_f':
            best100 = sorted(all_mols, key=lambda item: item.total, reverse=True)[:100]
            best10 = np.random.choice(best100,size=10,replace=False)
        elif experience_type == 'hvc':
            best100,fronts = nsga2_selection(all_mols,pop_size=100,return_fronts=True)
            if len(fronts[0])<=10:
                best10 = [all_mols[i] for i in fronts[0]]
            else:
                tmpidx = fronts[0]
                points = []
                scores = []
                for idx in tmpidx:
                    scores.append(all_mols[idx].scores)
                    points.append(all_mols[idx])
                scores = np.stack(scores)
                hv_pygmo = pg.hypervolume(scores)
                hvc = hv_pygmo.contributions(np.array([2.0 for i in range(scores.shape[1])]))

                sorted_indices = np.argsort(hvc)[::-1]  # Reverse to sort in descending order
                best10 = [points[i] for i in sorted_indices[:10]]
        elif experience_type == 'pareto':
            best100, fronts = nsga2_selection(all_mols,pop_size=100,return_fronts=True)
            best10 = np.random.choice(best100,size=10,replace=False)
        else:
            raise NotImplementedError("Not implemented experience type:", experience_type)

        requirement_prompt = self.make_requirement_prompt(self.original_mol,self.requirements,self.property)
        history_prompt = self.make_history_prompt(best10,experience=True)
        bad_history_prompt = self.make_history_prompt(worst10,experience=True)
        prompt = (
            f"I am optimizing molecular properties based on the following requirements:\n{requirement_prompt}\n\n"
            f"Here are some excellent non-dominated molecules and their associated property values:\n{history_prompt}\n\n"
            f"Here are some poorly performing molecules and their associated property values:\n{bad_history_prompt}\n\n"
            "Please analyze the patterns and characteristics of the excellent molecules. Summarize what makes them excel and suggest strategies to create new molecules with similar and better properties. "
            "Additionally, identify the reasons why the poorly performing molecules have suboptimal properties and provide guidance on how to avoid such issues in the future. "
        )
        
        if self.experience is not None:
            prompt += (
                f"When you write the experience, also integrate the important information from my old experience. \n"
                f"My old experience is: <old experience>{self.pure_experience} </old experience>\n"
                "In your reply, only write the concise integrated experience without repetitive sentences or "
                "conclusions to include as much useful experience as possible.\n"
                f"You can also abandon less useful or inappropriate experience in my old experience. "   
            )
            #print('-------------------')
            #print('combine experience query prompt:\n',prompt)
            #assert False
        prompt += ("Keep the summary concise (within 200 words), focusing on actionable insights and avoiding redundancy."
                   "Don't describe the given molecules, directly state the experience")

        self.exp_times += 1

        history_prompt = self.make_history_prompt(best10[:5],experience=True)
        bad_history_prompt = self.make_history_prompt(worst10[:5],experience=True)
        return prompt,history_prompt,bad_history_prompt


    def make_requirement_prompt(self,original_mol,requirements,properties):
        sentences = []
        number = 1
        
        for property in properties:
            if property == 'similarity':
                sentence = f"make sure the new molecules you propose has a similarity of over 0.4 to our original molecule <mol>{original_mol.value}</mol>"
            else:
                value = requirements[property+'_requ']
                property_name = value["property"]
                requirement = value["requirement"]

                # Check for specific requirement patterns directly using symbols
                if "increase" in requirement:
                    if ">=" in requirement:
                        threshold = requirement.split(">=")[-1].strip()
                        sentence = f"increase the {property_name} value by at least {threshold}."
                    elif ">" in requirement:
                        threshold = requirement.split(">")[-1].strip()
                        sentence = f"increase the {property_name} value to more than {threshold}."
                    else:
                        sentence = f"increase the {property_name} value."

                elif "decrease" in requirement:
                    if "<=" in requirement:
                        threshold = requirement.split("<=")[-1].strip()
                        sentence = f"decrease the {property_name} value to at most {threshold}."
                    elif "<" in requirement:
                        threshold = requirement.split("<")[-1].strip()
                        sentence = f"decrease the {property_name} value to less than {threshold}."
                    else:
                        sentence = f"decrease the {property_name} value."

                elif "range" in requirement:
                    # Extract the range values from the string
                    range_values = requirement.split(",")[1:]
                    range_start = range_values[0].strip()
                    range_end = range_values[1].strip()
                    sentence = f"keep the {property_name} value within the range {range_start} to {range_end}."
                elif "equal" in requirement:
                    equal_value = requirement.split(",")[1]
                    sentence = f"make sure {property_name} equals {equal_value}."
                elif "towards" in requirement:
                    equal_value = requirement.split(",")[1]
                    sentence = f"make sure {property_name} is towards {equal_value}."
                elif "the same" in requirement:
                    sentence = f"keep the {property_name} value the same."
                elif any(op in requirement for op in [">=", "<=", "=", ">", "<"]):
                    # Directly use the symbols for constraints
                    sentence = f"ensure the {property_name} value is {requirement}."
                else:
                    sentence = f"modify the {property_name} value."
            sentences.append(f'{number}. '+sentence)
            number += 1
        #init_sentence = f'Based on molecule <mol>{original_mol.value}</mol>, its property values are: '
        #for k,v in original_mol.property.items():
        #    init_sentence += f'{k}:{v:.4f}, '
        #sentences = init_sentence + f'suggest new molecules that satisfy the following requirements: \n' + '\n'.join(sentences) +'\n'
        sentences = f'suggest new molecules that satisfy the following requirements: \n' + '\n'.join(sentences) +'\n'
        if 'reduction_potential' in properties:
            sentences += 'reduction_potential is the most important objective, make sure it is as close to -1.3 as possible.'
        return sentences
import json
with open('data/descriptions.json','r') as f:
    descriptions = json.load(f) 

metadata = {
    "qed_requ": {
        "source_smiles": "O=C([C@H]1CCCC[C@H]1N1CCCC1=O)N1CC2(CC(F)C2)C1",
        "reference_smiles": "NNC(=O)C(=O)NC1CC2(C1)CN(C(=O)[C@H]1CCCC[C@H]1N1CCCC1=O)C2",
        "property": "QED",
        "requirement": "decrease <=2"
    },
    "logp_requ": {
        "source_smiles": "CCCCC(CC)COCOc1ccc([C@@H](O)C(=O)N[C@@H]2[C@H]3COC[C@@H]2CN(C(=O)CCc2ccccc2Cl)C3)cc1",
        "reference_smiles": "COc1ccc([C@@H](O)C(=O)N[C@H]2[C@@H]3COC[C@H]2CN(C(=O)CCc2ccccc2Cl)C3)cc1",
        "property": "logP",
        "requirement": "range, 2, 3"
    },
    "donor_requ": {
        "source_smiles": "O=C(NC[C@H]1CCOc2ccccc21)c1ccc(F)c(C(F)(F)F)c1",
        "reference_smiles": "CC(C)C[NH+](CC(=O)[O-])C(F)(F)c1cc(C(=O)NC[C@H]2CCOc3ccccc32)ccc1F",
        "property": "donor",
        "requirement": "increase >= 1"
    },
}

#Generate sentences based on metadata
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
#from algorithm.base import Item
if __name__ == '__main__':
    
    ops = ['qed','logp','donor','similarity']
    parents = [Item('CCFF',ops),Item('FFFFA',ops)]
    parents[0].raw_scores = [0,1,2,3]
    parents[1].raw_scores = [4,5,6,7]
    parents[0].assign_raw_scores([0,1,2,3])
    p = Prompt(parents[0],metadata,ops)
    prompt = p.get_crossover_prompt(parents)
    print(prompt)
