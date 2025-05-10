import numpy as np
import random

def mutation_single_point(offspring, ga_instance):
    for i in range(offspring.shape[0]):
        gene_idx = random.randint(0, offspring.shape[1] - 1)
        offspring[i, gene_idx] = 1 - offspring[i, gene_idx]
    return offspring

def mutation_two_point(offspring, ga_instance):
    for i in range(offspring.shape[0]):
        idx1, idx2 = sorted(random.sample(range(offspring.shape[1]), 2))
        offspring[i, idx1] = 1 - offspring[i, idx1]
        offspring[i, idx2] = 1 - offspring[i, idx2]
    return offspring

def mutation_edge(offspring, ga_instance):
    for i in range(offspring.shape[0]):
        offspring[i, 0] = 1 - offspring[i, 0]           # pierwszy bit
        offspring[i, -1] = 1 - offspring[i, -1]         # ostatni bit
    return offspring
