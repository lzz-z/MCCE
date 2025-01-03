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