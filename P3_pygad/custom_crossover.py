import numpy as np
import random

def crossover_arithmetic_func(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0], :]
        p2 = parents[(k + 1) % parents.shape[0], :]
        offspring[k, :] = 0.5 * p1 + 0.5 * p2
    return offspring

def crossover_linear_func(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0], :]
        p2 = parents[(k + 1) % parents.shape[0], :]

        c1 = 0.5 * p1 + 0.5 * p2
        c2 = 1.5 * p1 - 0.5 * p2
        c3 = -0.5 * p1 + 1.5 * p2

        # wybieramy jedno dziecko losowo z 3
        children = [c1, c2, c3]
        offspring[k, :] = random.choice(children)
    return offspring

def crossover_alpha_func(parents, offspring_size, ga_instance, alpha=0.5):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0], :]
        p2 = parents[(k + 1) % parents.shape[0], :]
        offspring[k, :] = alpha * p1 + (1 - alpha) * p2
    return offspring

def crossover_alpha_beta_func(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0], :]
        p2 = parents[(k + 1) % parents.shape[0], :]
        alpha = random.uniform(0, 1)
        beta = random.uniform(0, 1)
        offspring[k, :] = alpha * p1 + beta * p2
    return offspring

def crossover_average_func(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        p1 = parents[k % parents.shape[0], :]
        p2 = parents[(k + 1) % parents.shape[0], :]
        offspring[k, :] = (p1 + p2) / 2
    return offspring
