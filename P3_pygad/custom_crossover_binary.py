import numpy as np
import random

def crossover_single_point(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size, dtype=int)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k + 1) % parents.shape[0]]
        point = random.randint(1, len(p1) - 1)
        offspring[k, 0:point] = p1[0:point]
        offspring[k, point:] = p2[point:]
    return offspring

def crossover_two_point(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size, dtype=int)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k + 1) % parents.shape[0]]
        pt1, pt2 = sorted(random.sample(range(1, len(p1) - 1), 2))
        child = np.copy(p1)
        child[pt1:pt2] = p2[pt1:pt2]
        offspring[k] = child
    return offspring
