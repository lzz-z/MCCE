import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import pygmo as pg
import re

def extract_smiles_from_string(text):
    pattern = r"<candidate>(.*?)</candidate>"
    smiles_list = re.findall(pattern, text,flags=re.DOTALL)
    return smiles_list

def split_list(lst, n):
    """Splits the list lst into n nearly equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


def fast_non_dominated_sort(population):
    S = [[] for _ in range(len(population))]
    front = [[]]
    n = [0 for _ in range(len(population))]
    rank = [0 for _ in range(len(population))]

    for p in range(len(population)):
        S[p] = []
        n[p] = 0
        for q in range(len(population)):
            if dominates(population[p], population[q]):
                S[p].append(q) 
            elif dominates(population[q], population[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            front[0].append(p)
    
    i = 0
    while len(front[i]) != 0:
        Q = []
        for p in front[i]: # p: non dominated
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i = i + 1
        front.append(Q)

    del front[-1]
    return front

def dominates(ind1, ind2):
    not_worse_in_all = True
    strictly_better_in_one = False

    for x, y in zip(ind1.scores, ind2.scores):
        if x > y:
            not_worse_in_all = False
        if x < y:
            strictly_better_in_one = True

    return not_worse_in_all and strictly_better_in_one

def crowding_distance_assignment(front, population):
    distances = [0] * len(front)
    num_objectives = len(population[0].scores)
    
    for m in range(num_objectives):
        front.sort(key=lambda x: population[x].scores[m])
        distances[0] = distances[-1] = float('inf')
        for i in range(1, len(front) - 1):
            distances[i] += (population[front[i + 1]].scores[m] - population[front[i - 1]].scores[m]) / (max(population[k].scores[m] for k in front) - min(population[k].scores[m] for k in front))

    return distances

def nsga2_selection(population, pop_size,return_fronts=False):
    fronts = fast_non_dominated_sort(population)
    new_population = []
    for front in fronts:
        if len(new_population) + len(front) > pop_size:
            crowding_distances = crowding_distance_assignment(front, population)
            sorted_front = sorted(front, key=lambda x: crowding_distances[front.index(x)], reverse=True)
            new_population.extend(sorted_front[:pop_size - len(new_population)])
        else:
            new_population.extend(front)
    if return_fronts:
        return [population[i] for i in new_population],fronts
    return [population[i] for i in new_population]

def so_selection(population, pop_size):
    # Single objective
    sorted_items = sorted(population, key=lambda item: item.total, reverse=True)[:pop_size]
    return sorted_items

def nsga2_so_selection(population, pop_size):
    half_size = pop_size//2
    next_pops = so_selection(population,half_size)
    current_smis = [i.value for i in next_pops]
    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        candidates = [population[i] for i in front]
        candidates = sorted(candidates, key=lambda item: item.total, reverse=True)
        for can in candidates:
            if len(next_pops) >= pop_size:
                assert len(next_pops) == pop_size
                return next_pops
            if can.value not in current_smis:
                next_pops.append(can)
                current_smis.append(can.value)
    return next_pops
            
def hvc_selection(pops,pop_size):
    scores = []
    for pop in pops:
        scores.append(pop.scores)
    scores = np.stack(scores)
    hv_pygmo = pg.hypervolume(scores)
    hvc = hv_pygmo.contributions(np.array([1.1 for i in range(scores.shape[1])]))
    sorted_indices = np.argsort(hvc)[::-1]  # Reverse to sort in descending order
    bestn = [pops[i] for i in sorted_indices[:pop_size]]
    return bestn


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer, key=lambda kv: kv[1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[0].total, reverse=True))[:top_n]
        top_n_now = np.mean([item[0].total for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[0].total, reverse=True))[:top_n]
    top_n_now = np.mean([item[0].total for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


def cal_hv(scores):
    ref_point = np.array([1.1]*len(scores[0]))
    hv = HV(ref_point=ref_point)
    nds = NonDominatedSorting().do(scores,only_non_dominated_front=True)
    scores = scores[nds]
    return hv(scores)

