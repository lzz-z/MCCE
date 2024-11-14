from model.util import nsga2_selection
import pygmo as pg
import numpy as np
class Prompt:
    def __init__(self,original_mol,requirements,properties):
        self.requirements = requirements
        self.property = properties
        self.original_mol = original_mol
        self.experience = None

    def get_first_prompt(self):
        pass

    def get_final_prompt(self):
        pass 
    
    def get_mutation_prompt(self,ind_list,history_moles):
        requirement_prompt = self.make_requirement_prompt(self.original_mol,self.requirements,self.property)
        
        history_prompt = self.make_history_prompt(ind_list[:1])
        instruction_prompt = self.make_instruction_prompt(mutation=True)
        description_prompt = self.make_description_prompt()
        final_prompt =requirement_prompt + description_prompt +  history_prompt +  instruction_prompt 
        return final_prompt

    def get_crossover_prompt(self,ind_list,history_moles):
        requirement_prompt = self.make_requirement_prompt(self.original_mol,self.requirements,self.property)
        
        history_prompt = self.make_history_prompt(ind_list)
        instruction_prompt = self.make_instruction_prompt()
        description_prompt = self.make_description_prompt()
        mole_history_prompt = self.make_mole_history_prompt(history_moles)
        if self.experience is not None:
            final_prompt =requirement_prompt + description_prompt +  history_prompt + self.experience + instruction_prompt
        else:
            final_prompt =requirement_prompt + description_prompt +  history_prompt +  instruction_prompt
        #final_prompt =requirement_prompt  +  history_prompt +  instruction_prompt 
        return final_prompt

    def make_mole_history_prompt(self,moles):
        if len(moles) == 0 :
            return ""
        sentence = "Make sure you don't propose the following molecules: \n"
        for mole in moles:
            sentence += f"<mol>{mole}</mol>  "
        return sentence + '\n'

    def make_description_prompt(self):
        prompt = ""
        for p in self.property:
            prompt += p + ': ' + descriptions[p] + '\n'
        return prompt

    def make_history_prompt(self, ind_list, experience=False): # parent
        if experience:
            pop_content = ""
        else:
            pop_content = "I have some molecules with their objective values. \n"
        for ind in ind_list:
            pop_content += f"<mol>{ind.value}</mol>, its property values are: "
            for index,property in enumerate(ind.property_list):
                pop_content += f"{property}:{ind.raw_scores[index]:.4f},  "
            pop_content += f'total: {ind.total:.4f}\n'
        return pop_content
    
    def make_instruction_prompt(self,mutation=False): # improvement score = point hypervolume
        if mutation:
            prompt = ("Give me 2 new/novel better molecules that are different from all points above, and not dominated by any of the above. \n"
            "You can do it by applying mutation on the given points and based on your knowledge. The molecule should be valid. \n"
            "Do not write code. Do not give any explanation. Each output new molecule must start with <mol> and end with </mol> in SIMLE form"
            )
        else:
            prompt = ("Give me 2 new better molecules that are different from all points above, and not dominated by any of the above. \n"
            "You can do it by applying crossover on the given points and based on your knowledge. The molecule should be valid. \n"
            "Do not write code. Do not give any explanation. Each output new molecule must start with <mol> and end with </mol> in SIMLE form"
            )
        return prompt
    
    def make_experience_prompt(self,all_mols):
        best10 = sorted(all_mols, key=lambda item: item.total, reverse=True)[:10]
        worst10 = sorted(all_mols, key=lambda item: item.total, reverse=True)[-10:]
        best10, fronts = nsga2_selection(all_mols,pop_size=10,return_fronts=True)
        '''best100,fronts = nsga2_selection(all_mols,pop_size=100,return_fronts=True)
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
            hvc = hv_pygmo.contributions(np.array([1.0 for i in range(scores.shape[1])]))

            sorted_indices = np.argsort(hvc)[::-1]  # Reverse to sort in descending order
            best10 = [points[i] for i in sorted_indices[:10]]'''

        requirement_prompt = self.make_requirement_prompt(self.original_mol,self.requirements,self.property)
        history_prompt = self.make_history_prompt(best10,experience=True)
        bad_history_prompt = self.make_history_prompt(worst10,experience=True)
        prompt = "I am optimizing molecular properties, the requirements are in the following: \n" + requirement_prompt + "\n"
        prompt = prompt + "I have already obtained some excellent non-dominated molecules with its property values:" + history_prompt + "\n"
        prompt = prompt + "I also have some worst molecules with its property values:" + history_prompt + "\n"
        prompt = prompt + "now summarize the experience of the excellence of these molecules and how can we create novel molecules that are excellent like these."
        prompt = prompt + "And also summarize the why molecules have bad propeties and how to avoid them."
        prompt = prompt + "Keep the experience in 400 words, don't be too long and repetitive."
        return prompt,history_prompt


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
        init_sentence = f'Based on molecule <mol>{original_mol.value}</mol>, its property values are: '
        for k,v in original_mol.property.items():
            init_sentence += f'{k}:{v:.4f}, '
        #sentences = init_sentence + f'suggest new molecules that satisfy the following requirements: \n' + '\n'.join(sentences) +'\n'
        sentences = f'suggest new molecules that satisfy the following requirements: \n' + '\n'.join(sentences) +'\n'
        return sentences

descriptions = {
    "qed":("QED (Quantitative Estimate of Drug-likeness) is a measure that quantifies"
        "how 'drug-like' a molecule is based on properties such as molecular weight,"
            "solubility, and the number of hydrogen bond donors and acceptors."  
            "Adding functional groups that improve drug-like properties (e.g., small molecular size,"
            "balanced hydrophilicity) can increase QED, while introducing large, complex, or highly polar groups can decrease it."),

    "logp":("LogP is the logarithm of the partition coefficient, measuring the lipophilicity"
          "or hydrophobicity of a molecule, indicating its solubility in fats versus water."
          "Adding hydrophobic groups (e.g., alkyl chains or aromatic rings) increases LogP,"
            "while adding polar or hydrophilic groups (e.g., hydroxyl or carboxyl groups) decreases it."), 

    "donor":("Donor Number refers to the number of hydrogen bond donors (atoms like NH or OH) in a molecule,"
        "influencing its interaction with biological targets." 
        "Introducing additional hydrogen bond donors (e.g., hydroxyl or amine groups) increases"
              "the Donor Number, while removing or modifying these groups decreases it."),

    "similarity":("Similarity in this context is calculated using Morgan fingerprints, which represent molecular"
        "structures, and Tanimoto similarity measures how structurally similar two molecules are based on their fingerprints."
        "Modifying the core structure of a molecule significantly (e.g., ring opening or closing) decreases similarity,"
            "while smaller changes like side-chain substitutions tend to have a lesser impact on similarity."),
    "logs":(
        "Log S indicates the solubility of a molecule in water, with higher values showing better solubility. "
        "Adding polar functional groups (like -OH or -COOH) can increase Log S, while adding hydrophobic groups "
        "(like long alkyl chains) can decrease it."
    ),
    "reduction_potential":(
        "Reduction potential quantifies a molecule's tendency to gain electrons and undergo reduction. Introducing "
        "electron-withdrawing groups (like -NO2) can increase reduction potential, while adding electron-donating groups "
        "(like -OH or -CH3) can decrease it."
    ),
    "sa":(
        "SA measures how easily a molecule can be synthesized based on its structural complexity. Simplifying "
        "a molecule by reducing complex ring systems or functional groups can lower SA, making synthesis easier, "
        "while adding complex structures can increase SA, making synthesis harder."
    ),
    "drd2":(
        "Dopamine receptor D2 (DRD2) is a receptor involved in the modulation of neurotransmission and is a target for various psychiatric and neurological disorders. "
        "Adding functional groups like hydroxyl or halogen atoms to aromatic rings can enhance binding affinity to DRD2. "
        "Removing aromaticity or introducing bulky groups near the binding sites often decreases DRD2 activity."
    ),
    "gsk3b":(
        "Glycogen synthase kinase-3 beta (GSK3β) is an enzyme involved in cellular processes like metabolism and apoptosis, and is a therapeutic target for cancer and neurological diseases."
        "Adding polar groups, such as hydroxyls, can improve hydrogen bonding with GSK3β's active site."
        "Introducing steric hindrance or highly hydrophobic regions can reduce interactions with GSK3β."
    ),
    "jnk3":(
        "c-Jun N-terminal kinase 3 (JNK3) is a kinase involved in stress signaling and is targeted for neuroprotection in diseases like Alzheimer's."
        "Introducing small polar or electronegative groups can enhance binding affinity to JNK3."
        "Removing polar functional groups or adding large, bulky substituents can reduce activity by obstructing the active site."
    ),
    "smarts_filter":(
        "To pass the SMARTS filter, the proposed molecule must not have the following substructures:"
        "reactive alkyl halides: [Br,Cl,I][CX4;CH,CH2]"
        "acid halides: [S,C](=[O,S])[F,Br,Cl,I]"
        "carbazides: O=CN=[N+]=[N-]"
        "sulphate esters: COS(=O)O[C,c]"
        "sulphonates: COS(=O)(=O)[C,c]"
        "acid anhydrides: C(=O)OC(=O)"
        "peroxides: OO"
        "pentafluorophenyl esters: C(=O)Oc1c(F)c(F)c(F)c(F)c1(F)"
        "esters of HOBT: C(=O)Onnn"
        "isocyanates & isothiocyanates: N=C=[S,O]"
        "triflates: OS(=O)(=O)C(F)(F)F"
        "lawesson's reagent and derivatives: P(=S)(S)S"
        "phosphoramides: NP(=O)(N)N"
        "aromatic azides: cN=[N+]=[N-]"
        "acylhydrazide: [N;R0][N;R0]C(=O)"
        "quaternary C, Cl, I, P or S: [C+,Cl+,I+,P+,S+]"
        "phosphoranes: C=P"
        "chloramidines: [Cl]C([C&R0])=N"
        "nitroso: [N&D2](=O)"
        "P/S Halides: [P,S][Cl,Br,F,I]"
        "carbodiimide: N=C=N"
        "isonitrile: [N+]#[C-]"
        "triacyloximes: C(=O)N(C(=O))OC(=O)"
        "cyanohydrins: N#CC[OH]"
        "acyl cyanides: N#CC(=O)"
        "sulfonyl cyanides: S(=O)(=O)C#N"
        "cyanophosphonates: P(OCC)(OCC)(=O)C#N"
        "azocyanamides: [N;R0]=[N;R0]C#N"
        "azoalkanals: [N;R0]=[N;R0]CC=O"
        "epoxides, thioepoxides, aziridines: C1[O,S,N]C1"
        "esters, thioesters: C[O,S;R0][C;R0](=[O,S])"
        "cyanamides: NC#N"
        "four membered lactones: C1(=O)OCC1"
        "betalactams: N1CCC1=O"
        "di and triphosphates: P(=O)([OH])OP(=O)[OH]"
        "acyclic C=C-O: C=[C!r]O"
        "amidotetrazole: c1nnnn1C=O"
        "azo group: N#N"
        "hydroxamic acid: C(=O)N[OH]"
        "imine: C=[N!R]"
        "imine: N=[CR0][N,n,O,S]"
        "ketene: C=C=O"
        "nitro group: [N+](=O)[O-]"
        "N-nitroso: [#7]-N=O"
        "oxime: [C,c]=N[OH]"
        "oxime: [C,c]=NOC=O"
        "Oxygen-nitrogen single bond: [OR0,NR0][OR0,NR0]"
        "perfluorinated chain: [CX4](F)(F)[CX4](F)F"
    )
}

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
