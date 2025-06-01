import numpy as np
from mealpy.swarm_based.ACOR import OriginalACOR
from mealpy.utils.problem import Problem
from mealpy.utils.space import FloatVar
import matplotlib.pyplot as plt
import pandas as pd
import time

# Definicje funkcji benchmarkowych

def hypersphere(x):
    return sum([xi**2 for xi in x])

def rana(x):
    # Implementacja funkcji Rana dla n-wymiarów, minimalizacja
    # https://www.sfu.ca/~ssurjano/rana.html
    total = 0
    n = len(x)
    for i in range(n - 1):
        xi = x[i]
        xnext = x[i + 1]
        term1 = (xi * np.sin(np.sqrt(abs(xnext + 1 - xi))) * np.cos(np.sqrt(abs(xnext + 1 + xi))))
        term2 = ((xnext + 1) * np.cos(np.sqrt(abs(xnext + 1 - xi))) * np.sin(np.sqrt(abs(xnext + 1 + xi))))
        total += term1 + term2
    return total

# --- KONFIGURACJA ---

config = {
    "algorithm": OriginalACOR,
    "dimensions": 2,
    "precision": 6,
    "population": 100,
    "generations": 100,
    "benchmark_function": "rana",  # "hypersphere" lub "rana"
    "lower_bound": -512,
    "upper_bound": 512,
    "runs": 5,
    "algo_name": "BaseACOR"
}

# Wybór funkcji celu na podstawie configu
if config["benchmark_function"] == "hypersphere":
    fitness_function = hypersphere
    config["lower_bound"] = -100
    config["upper_bound"] = 100
elif config["benchmark_function"] == "rana":
    fitness_function = rana
    config["lower_bound"] = -500
    config["upper_bound"] = 500
else:
    raise ValueError("Nieznana funkcja benchmarkowa")

# --- GŁÓWNA PĘTLA ---

all_bests, all_means, all_stds = [], [], []
fitness_scores, execution_times, best_solutions = [], [], []

for run in range(config["runs"]):
    print(f"Run {run + 1}/{config['runs']}...")

    def fitness(solution):
        rounded = np.round(solution, config["precision"])
        return fitness_function(rounded)

    bounds = [FloatVar(lb=config["lower_bound"], ub=config["upper_bound"]) for _ in range(config["dimensions"])]

    problem = Problem(
        obj_func=fitness,
        bounds=bounds,
        minmax="min"
    )

    model = OriginalACOR(
        problem=problem,
        epoch=config["generations"],
        pop_size=config["population"],
        verbose=False
    )

    start_time = time.time()
    best_agent = model.solve(problem=problem)
    exec_time = time.time() - start_time

    print(f"Best agent solution: {best_agent.solution}")
    print(f"Best agent fitness: {best_agent.target.fitness}")

    fitness_scores.append(best_agent.target.fitness)
    execution_times.append(exec_time)
    best_solutions.append(best_agent.solution)

    if hasattr(model, 'history') and hasattr(model.history, 'list_global_best_fit'):
        all_bests.append(model.history.list_global_best_fit)
    else:
        all_bests.append([])

# Podsumowanie
best_result = min(fitness_scores)
worst_result = max(fitness_scores)
avg_result = np.mean(fitness_scores)
avg_time = np.mean(execution_times)
best_index = fitness_scores.index(best_result)

print(f"\n{config['algo_name']} on {config['benchmark_function']}:")
print(f"Best: {best_result}")
print(f"Worst: {worst_result}")
print(f"Avg: {avg_result}")
print(f"Avg Time: {avg_time:.2f}s")
print(f"Best Solution: {np.round(best_solutions[best_index], config['precision'])}")

# Wykres, jeśli dostępne dane z historii
if all_bests[best_index]:
    plt.figure(figsize=(10, 6))
    plt.plot(all_bests[best_index], label='Best Fitness')
    plt.xlabel("Iteracja")
    plt.ylabel("Wartość funkcji celu")
    plt.title(f"Najlepsza wartość funkcji celu w kolejnych iteracjach - {config['benchmark_function']}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
