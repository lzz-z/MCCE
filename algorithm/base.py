import numpy as np
import pickle
import os
from typing import List, Dict, Any

class Item:
    def __init__(self, value: str, property_list: List[str]) -> None:
        """
        Initializes an Item object representing a molecule with specified properties.

        Args:
            value (str): The molecule's SMILES string or identifier.
            property_list (List[str]): List of property names to be assigned to the molecule.
        """
        self.value = value
        self.property_list = property_list
        self.total = None
        self.scores = None
        self.property = None

    def assign_results(self,results:Dict):
        self.property = results['original_results']
        self.scores = [results['transformed_results'][obj] for obj in self.property_list]
        self.total = results['overall_score']

class ItemFactory:
    def __init__(self, property_list: List[str]) -> None:
        """
        Factory class to generate Item instances with a predefined property_list.

        Args:
            property_list (List[str]): The list of property names to assign to each Item.
        """
        self.property_list = property_list

    def create(self, value: str) -> Item:
        """
        Create a new Item with the factory's property_list.

        Args:
            value (str): The molecule's SMILES string or identifier.

        Returns:
            Item: A new Item instance.
        """
        return Item(value, self.property_list)
    
class HistoryBuffer:
    def __init__(self) -> None:
        """
        Initializes a history buffer for storing prompts, generations, and responses.
        """
        self.prompts = []
        self.generations= []
        self.responses = []
        self.successful_molecules = []
        self.failed_molecules = []
        self.save_path = 'checkpoint/'

    def save_to_pkl(self, filename: str) -> None:
        """
        Saves the entire buffer object to a pickle file.

        Args:
            filename (str): The name of the output pickle file.
        """
        os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, filename), 'wb') as f:
            pickle.dump(self, f)
        print(f"Data saved to {filename}")

    def load_from_pkl(self, filename: str) -> 'HistoryBuffer':
        """
        Loads a HistoryBuffer object from a pickle file.

        Args:
            filename (str): The pickle file name to load from.

        Returns:
            HistoryBuffer: Loaded history buffer instance.
        """
        with open(os.path.join(self.save_path, filename), 'rb') as f:
            obj = pickle.load(f)
        print(f"Data loaded from {filename}")
        return obj

    def push(self, prompts: Any, generation: Any, responses: Any) -> None:
        """
        Appends a new generation record to the buffer.

        Args:
            prompts (Any): The prompt(s) used to generate molecules.
            generation (Any): Generated molecule(s).
            responses (Any): Model's response(s), e.g., SMILES strings.
        """
        self.prompts.append(prompts)
        self.generations.append(generation)
        self.responses.append(responses)


'''
    'celecoxib_rediscovery', 'troglitazone_rediscovery','thiothixene_rediscovery',
'albuterol_similarity','mestranol_similarity',
'isomers_c7h8n2o2','isomers_c9h10n2o2pf2cl','median1','median2', 'osimertinib_mpo',
'fexofenadine_mpo','ranolazine_mpo','perindopril_mpo', 'amlodipine_mpo',
'sitagliptin_mpo','zaleplon_mpo','valsartan_smarts', 'deco_hop', 'scaffold_hop'
'''