from model.util import nsga2_selection
import numpy as np

from algorithm.base import Item
import pygmo as pg
from typing import List, Dict, Optional, Union
import json
import random
import yaml


class Prompt:
    def __init__(self, config, original_item: Item = None):
        """
        Generic Prompt class for multi-objective optimization tasks using LLMs.

        Args:
            config (Config): Configuration object containing task parameters.
            original_item (Item): The original Item used as a reference (optional).
        """
        self.original_item = original_item
        self.config = config
        self.requirements = config.get('optimization_direction')
        self.properties = config.get('goals')
        self.obj_directions = {obj: config.get('optimization_direction')[i] for i, obj in enumerate(self.properties)}
        self.experience = None
        self.pure_experience = None
        self.exp_times = 0
        self.experience_prob = config.get('model.experience_prob')
        self.num_offspring = self.config.get('num_offspring',default=2)
        with open(config.get("prompt_info_path"), "r") as yaml_file:
            self.info = yaml.safe_load(yaml_file)
            if self.config.get('n_circles',default=False):
                n_circles = self.config.get('n_circles')
                self.info['description'] = self.info['description'].replace('32 circles',f'{n_circles} circles') 

    def get_prompt(self, prompt_type: str, ind_list: List[Item], history_items: List[Item]) -> str:
        experience = self.experience if self.experience and np.random.random() < self.experience_prob else ""
        if prompt_type == 'crossover':
            return self._get_crossover_prompt(ind_list, history_items, experience)
        elif prompt_type == 'mutation':
            return self._get_mutation_prompt(ind_list, history_items, experience)
        elif prompt_type == 'explore':
            return self._get_exploration_prompt(history_items)
        else:
            raise NotImplementedError(f'Unsupported operation type: {prompt_type}')

    def _get_crossover_prompt(self, ind_list: List[Item], history_items: List[Item], experience: str) -> str:
        return self._compose_prompt(ind_list, experience, 'crossover')

    def _get_mutation_prompt(self, ind_list: List[Item], history_items: List[Item], experience: str) -> str:
        return self._compose_prompt(ind_list, experience, 'mutation')

    def _get_exploration_prompt(self, history_items: List[Item]) -> str:
        top100 = sorted(history_items, key=lambda x: x.total, reverse=True)[:100]
        worst10 = sorted(history_items, key=lambda x: x.total)[self.exp_times * 10:(self.exp_times + 1) * 10]
        random10 = np.random.choice(top100, size=10, replace=False)

        prompt = self._compose_prompt(random10, "", 'explore')
        prompt += "There are also some bad candidates, don't propose new candidates like the candidates below:\n"
        prompt += self._make_history_prompt(worst10, experience=False)
        return prompt

    def _compose_prompt(self, ind_list: List[Item], experience: str, oper_type: str) -> str:
        parts = [
            self.info['description'],
            self._make_requirement_prompt(),
            self._make_description_prompt(),
            experience,
            self._make_history_prompt(ind_list),
            self._make_instruction_prompt(oper_type)
        ]
        final_prompt = ''.join(parts)
        #print('prompt example:',final_prompt)
        #assert False
        return final_prompt

    def _make_description_prompt(self) -> str:
        return ''.join([f"{p}: {self.info[p]}\n" for p in self.properties])

    def _make_history_prompt(self, ind_list: List[Item], experience: bool = False) -> str:
        header = "" if experience else (
            "I have some candidates with their objective values. The total score is the integration of all property values; a higher total score means a better candidate.\n"
            "If the total score of the parent candidates is 0 or very low, you can discard them and repropose with your knowledge.")

        entries = []
        for ind in ind_list:
            entry = f"<candidate>{ind.value}</candidate>, its property values are: " + \
                    ", ".join([f"{prop}:{score:.4f}" for prop, score in ind.property.items()]) + \
                    f", total: {ind.total:.4f}"

            # 如果有 constraints，添加说明
            if ind.constraints is not None:
                constraint_str = ", ".join([f"{k}:{v}" for k, v in ind.constraints.items()])
                entry += f", constraint values are: {constraint_str}"
            entries.append(entry + "\n")

        return header + ''.join(entries)


    def _make_instruction_prompt(self, oper_type: str) -> str:
        common_tail = (self.info['other_requirements'] + "\n" 
                       + self.info['example_output'] + "\n" +
                        "Do not give any explanation." )

        if oper_type == 'mutation':
            return (
                f"Generate {self.num_offspring } new better candidates through mutation, ensuring they are different from all points provided above and not dominated by any of them.\n"
                + self.info['mutation_instruction'] + '\n' + common_tail)
        elif oper_type == 'crossover':
            return (
                f"Give me {self.num_offspring} new better candidates that are different from all points above and not dominated by them.\n"
                "Use crossover and your knowledge to create better candidates.\n"
                + self.info['crossover_instruction'] + '\n' + common_tail)
        elif oper_type == 'explore':
            return (
                f"Confidently propose {self.num_offspring } novel and better candidates different from the given ones, leveraging your expertise.\n" + common_tail)
        else:
            raise NotImplementedError(f'Unsupported instruction type: {oper_type}')

    def _make_requirement_prompt(self) -> str:
        sentences = [
            f"{i+1}. {self._translate_requirement(prop)}"
            for i, prop in enumerate(self.properties)
        ]
        header = " Suggest new candidates that satisfy the following requirements:\n"
        if 'reduction_potential' in self.properties:
            header += 'reduction_potential is the most important objective, make sure it is as close to -1.3 as possible.\n'
        return header + '\n'.join(sentences) + '\n'

    def _translate_requirement(self, prop: str) -> str:
        if self.obj_directions[prop] == 'max':
            return f"maximize the {prop} value."
        if self.obj_directions[prop] == 'min':
            return f"minimize the {prop} value."

    def make_experience_prompt(self, all_items: List[tuple]) -> tuple[str, str, str]:
        all_items = [i[0] for i in all_items]
        experience_type = np.random.choice(['best_f', 'hvc', 'pareto'], p=[0.5, 0., 0.5])
        sorted_items = sorted(all_items, key=lambda x: x.total)

        # 取后半部分
        half = len(sorted_items) // 2
        back_half = sorted_items[half:]
        if half<10:
            worst10 = random.choices(back_half, k=10)
        else:
            worst10 = random.sample(back_half, k=10)


        if experience_type == 'best_f':
            top100 = sorted(all_items, key=lambda x: x.total, reverse=True)[:100]
            best10 = list(np.random.choice(top100, size=10, replace=False))

        elif experience_type == 'pareto':
            best100, _ = nsga2_selection(all_items, pop_size=100, return_fronts=True)
            best10 = list(np.random.choice(best100, size=10, replace=False))

        elif experience_type == 'hvc':
            best100, fronts = nsga2_selection(all_items, pop_size=100, return_fronts=True)
            if len(fronts[0]) <= 10:
                best10 = [all_items[i] for i in fronts[0]]
            else:
                scores = np.array([all_items[i].scores for i in fronts[0]])
                points = [all_items[i] for i in fronts[0]]
                hv = pg.hypervolume(scores)
                contrib = hv.contributions(ref_point=np.array([2.0] * scores.shape[1]))
                sorted_idx = np.argsort(contrib)[::-1]
                best10 = [points[i] for i in sorted_idx[:10]]

        else:
            raise NotImplementedError(f"Unsupported experience type: {experience_type}")

        best_prompt = self._make_history_prompt(best10, experience=True)
        worst_prompt = self._make_history_prompt(worst10, experience=True)
        requirement_text = self._make_requirement_prompt()

        summary_prompt = (
            f"I am optimizing candidate properties based on the following requirements:\n{requirement_text}\n"
            f"Here are some excellent non-dominated candidates and their associated property values:\n{best_prompt}\n"
            f"Here are some poorly performing candidates and their associated property values:\n{worst_prompt}\n"
            "Please analyze the patterns and characteristics of the excellent candidates. Summarize what makes them excel and suggest strategies "
            "to create new candidates with similar and better properties. Additionally, identify the reasons why the poorly performing candidates "
            "have suboptimal properties and provide guidance on how to avoid such issues in the future.\n"
        )

        if self.experience:
            summary_prompt += (
                "When you write the experience, also integrate the important information from my old experience:\n"
                f"<old experience>{self.pure_experience}</old experience>\n"
                "In your reply, only write the concise integrated experience without repetitive sentences or conclusions.\n"
                "You can also abandon less useful or inappropriate experience in my old experience.\n"
            )

        summary_prompt += (
            "Keep the summary concise (within 200 words), focusing on actionable insights and avoiding redundancy. "
            "Don't describe the given candidates, directly state the experience."
        )

        self.exp_times += 1
        return summary_prompt, self._make_history_prompt(best10[:5], experience=True), self._make_history_prompt(worst10[:5], experience=True)



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
import json
if __name__ == '__main__':
    with open('/home/hp/src/MOLLM/data/goals5.json','r') as f:
        metadata = json.load(f)
    metadata = metadata['requirements'][0]
    ops = ['qed','sa','jnk3']
    parents = [Item('CCFF',ops),Item('FFFFA',ops)]
    parents[0].raw_scores = [0,1,2,3]
    parents[1].raw_scores = [4,5,6,7]
    parents[0].assign_raw_scores([0,1,2,3])
    p = Prompt(parents[0],metadata,ops)
    prompt = p.get_prompt( prompt_type='mutation', ind_list=parents, history_moles=parents)
    print(prompt)
