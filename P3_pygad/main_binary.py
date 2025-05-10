import pygad
import numpy as np
import matplotlib.pyplot as plt

from benchmark_setup import evaluate_fitness, get_dimensions, get_bounds

from custom_crossover_binary import (
    crossover_single_point,
    crossover_two_point
)

from custom_mutation_binary import (
    mutation_single_point,
    mutation_two_point,
    mutation_edge
)

# PARAMETRY Z FUNKCJI CELU
num_variables = get_dimensions()
bounds = get_bounds()
lower_bound, upper_bound = bounds[0]  # zakładamy, że zakres ten sam dla wszystkich zmiennych

# KONFIGURACJA BITÓW
bits_per_variable = 20
num_genes = num_variables * bits_per_variable
generation_best_fitness = []

# DEKODOWANIE OSOBNIKA BINARNEGO NA ZMIENNE FLOAT
def decodeInd(individual):
    decoded = []
    for i in range(num_variables):
        start = i * bits_per_variable
        end = start + bits_per_variable
        binary_string = ''.join(str(bit) for bit in individual[start:end])
        decimal_value = int(binary_string, 2)
        max_decimal = 2**bits_per_variable - 1
        real_value = lower_bound + (decimal_value / max_decimal) * (upper_bound - lower_bound)
        decoded.append(real_value)
    return decoded

# FUNKCJA CELU
def fitnessFunction(ga_instance, individual, solution_idx):
    real_vector = decodeInd(individual)
    return evaluate_fitness(real_vector)

# MONITORING POSTĘPU
def on_generation(ga):
    best = ga.best_solution()[1]
    generation_best_fitness.append(-best)

# URUCHOMIENIE ALGORYTMU
ga_instance = pygad.GA(
    num_generations=100,
    sol_per_pop=50,
    num_parents_mating=20,
    num_genes=num_genes,
    init_range_low=0,
    init_range_high=2,
    gene_type=int,
    fitness_func=fitnessFunction,
    parent_selection_type="tournament",
    crossover_type=crossover_single_point,     # ← możesz zmieniać
    mutation_type=mutation_edge,              # ← możesz zmieniać
    mutation_percent_genes=5,
    keep_elitism=2,
    on_generation=on_generation
)

ga_instance.run()

# NAJLEPSZY OSOBNIK
solution, solution_fitness, _ = ga_instance.best_solution()
decoded_solution = decodeInd(solution)
print("Najlepszy osobnik (binarne):", solution)
print("Dekodowany (rzeczywiste zmienne):", decoded_solution)
print("Wartość funkcji celu:", -solution_fitness)

# WYKRES
plt.plot(generation_best_fitness, label='Najlepszy wynik')
plt.xlabel("Generacja")
plt.ylabel("Wartość funkcji celu")
plt.title("Postęp optymalizacji (binarna reprezentacja, benchmark)")
plt.legend()
plt.grid(True)
plt.show()

