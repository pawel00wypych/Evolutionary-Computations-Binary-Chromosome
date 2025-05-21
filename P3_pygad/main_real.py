import pygad
import numpy as np
import matplotlib.pyplot as plt

from benchmark_setup import evaluate_fitness, get_dimensions

from custom_crossover import (
    crossover_arithmetic_func,
    crossover_linear_func,
    crossover_alpha_func,
    crossover_alpha_beta_func,
    crossover_average_func
)

from custom_mutation import (
    mutation_uniform_func,
    mutation_gaussian_func
)

# PARAMETRY KONFIGURACYJNE
num_variables = get_dimensions()
generation_best_fitness = []

# FUNKCJA CELU (benchmarkowa)
def fitnessFunction(ga_instance, individual, solution_idx):
    return evaluate_fitness(individual)

# MONITORING POSTĘPU
def on_generation(ga):
    best = ga.best_solution()[1]
    generation_best_fitness.append(-best)

# URUCHOMIENIE ALGORYTMU
ga_instance = pygad.GA(
    num_generations=100,
    sol_per_pop=50,
    num_parents_mating=20,
    num_genes=num_variables,
    init_range_low=-100,    # <- ogólny zakres, prawdziwy ustala benchmark_setup.py
    init_range_high=100,
    gene_type=float,
    fitness_func=fitnessFunction,
    parent_selection_type="tournament",
    crossover_type=crossover_average_func,       # <- wybierz inne jeśli chcesz
    mutation_type=mutation_gaussian_func,        # <- wybierz inne jeśli chcesz
    mutation_percent_genes=5,
    keep_elitism=2,
    on_generation=on_generation
)

ga_instance.run()

# NAJLEPSZY OSOBNIK
solution, solution_fitness, _ = ga_instance.best_solution()
print("Najlepszy osobnik (rzeczywiste zmienne):", solution)
print("Wartość funkcji celu:", -solution_fitness)

# WYKRES
plt.plot(generation_best_fitness, label='Najlepszy wynik')
plt.xlabel("Generacja")
plt.ylabel("Wartość funkcji celu")
plt.title("Postęp optymalizacji (benchmark)")
plt.legend()
plt.grid(True)
plt.show()
