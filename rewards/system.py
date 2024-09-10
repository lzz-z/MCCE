import requests
import json
url = 'http://cpu1.ms.wyue.site:8000/process'


def get_evaluation(evaluate_metric, smiles):
    data = {
        "ops": evaluate_metric,
        "data":smiles
    }
    response = requests.post(url, json=data)
    result = response.json()['results']
    return result

class Rewarding_system:
    def __init__(self):
        self.all_rewards = {}

    def register_reward(self, reward_name, reward_function):
        self.all_rewards[reward_name] = reward_function

    def get_reward(self, reward_name, items):
        return self.all_rewards[reward_name](*items)

    def evaluate(self,ops,smiles_list):
        return get_evaluation(ops,smiles_list)

    def evaluate_items(self, items, qed_requ, logp_requ, donor_requ, donor_num):
        smiles_list = [[item.value] for item in items]
        fitnesses, donors, logps, qeds = self._get_evaluation(smiles_list, qed_requ, logp_requ, donor_requ, donor_num)

        for i, item in enumerate(items):
            item.scores = {'qed': qeds[i], 'logp': logps[i], 'donor': donors[i]}
            item.fitness = fitnesses[i]

    def _get_evaluation(self, smiles_list, qed_requ, logp_requ, donor_requ, donor_num):
        # 具体的评估函数与之前的类似
        pass