import os
import csv
import time
from datetime import datetime

from genetic_algorithm.chromosome_real import ChromosomeReal
from genetic_algorithm.population_real import PopulationReal
from genetic_algorithm.crossover_real import CrossoverReal
from genetic_algorithm.mutation_real import MutationReal
from genetic_algorithm.evaluation_functions import hypersphere_fitness

# -------------------- PARAMETRY --------------------

POPULATION_SIZE = 100
EPOCHS = 100
MUTATION_PROB = 0.1
CROSSOVER_PROB = 0.7
VARIABLE_RANGES = [(-5, 5)] * 2  # 2 zmienne w przedziale [-5, 5]
NUM_OF_VARIABLES = len(VARIABLE_RANGES)

# -------------------- INICJALIZACJA --------------------

population = PopulationReal(NUM_OF_VARIABLES, VARIABLE_RANGES)
population.create_initial_population(POPULATION_SIZE)
population.evaluate(hypersphere_fitness)

# -------------------- STATYSTYKI --------------------

best_fitness_per_epoch = []
avg_fitness_per_epoch = []
std_fitness_per_epoch = []

# -------------------- EWOLUCJA --------------------

start_time = time.time()

for epoch in range(EPOCHS):
    # SELEKCJA turniejowa — bierzemy najlepszych (tu: sortujemy ręcznie)
    population.individuals.sort(key=lambda ind: ind.fitness)
    selected = population.individuals[:POPULATION_SIZE]

    # ELITARNE: zachowaj najlepszego
    elite = selected[0].clone()

    # KRZYŻOWANIE
    crossover = CrossoverReal(selected, CROSSOVER_PROB)
    offspring = crossover.arithmetic_crossover()

    # MUTACJA
    mutation = MutationReal(offspring, MUTATION_PROB, VARIABLE_RANGES)
    mutated = mutation.gaussian_mutation(sigma=0.5)

    # NOWA POPULACJA
    population.individuals = mutated
    population.individuals[0] = elite  # zachowujemy najlepszego

    # OCENA
    population.evaluate(hypersphere_fitness)

    # STATYSTYKI
    fitness_vals = [ind.fitness for ind in population.individuals]
    best = min(fitness_vals)
    avg = sum(fitness_vals) / len(fitness_vals)
    std = (sum((x - avg)**2 for x in fitness_vals) / len(fitness_vals)) ** 0.5

    best_fitness_per_epoch.append(best)
    avg_fitness_per_epoch.append(avg)
    std_fitness_per_epoch.append(std)

    print(f"Epoka {epoch+1}/{EPOCHS} | Best: {best:.6f} | Avg: {avg:.6f} | Std: {std:.6f}")

end_time = time.time()
print(f"\nCzas działania: {end_time - start_time:.2f} sekundy")

# -------------------- ZAPIS DO CSV --------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"results_real_{timestamp}.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "BestFitness", "AvgFitness", "StdDev"])
    for i in range(EPOCHS):
        writer.writerow([i+1, best_fitness_per_epoch[i], avg_fitness_per_epoch[i], std_fitness_per_epoch[i]])

print(f"Wyniki zapisane do pliku: {csv_file}")
