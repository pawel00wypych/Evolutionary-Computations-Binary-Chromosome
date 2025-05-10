import numpy as np
import random

def mutation_uniform_func(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        gene_idx = np.random.choice(range(offspring.shape[1]))
        min_val = ga_instance.init_range_low
        max_val = ga_instance.init_range_high
        offspring[chromosome_idx, gene_idx] = np.random.uniform(min_val, max_val)
    return offspring

def mutation_gaussian_func(offspring, ga_instance, sigma=1.0):
    for chromosome_idx in range(offspring.shape[0]):
        gene_idx = np.random.choice(range(offspring.shape[1]))
        current_val = offspring[chromosome_idx, gene_idx]
        mutated = current_val + np.random.normal(0, sigma)
        mutated = np.clip(mutated, ga_instance.init_range_low, ga_instance.init_range_high)
        offspring[chromosome_idx, gene_idx] = mutated
    return offspring
