import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from genetic_algorithm.chromosome_real import ChromosomeReal
from genetic_algorithm.population_real import PopulationReal
from genetic_algorithm.crossover_real import CrossoverReal
from genetic_algorithm.mutation_real import MutationReal
from genetic_algorithm.evaluation_functions import hypersphere_fitness, hybrid_fitness
import time
import csv
from datetime import datetime


def run_algorithm():
    # Pobranie parametrów z GUI
    epochs = int(entry_epochs.get())
    pop_size = int(entry_population.get())
    mutation_prob = float(entry_mutation.get())
    crossover_prob = float(entry_crossover.get())
    func_name = combo_function.get()
    crossover_method = combo_crossover.get()
    mutation_method = combo_mutation.get()

    # Wybór funkcji celu
    if func_name == "hypersphere":
        fitness_function = hypersphere_fitness
        variable_ranges = [(-5, 5)] * 2
    elif func_name == "cec_hybrid":
        fitness_function = hybrid_fitness
        variable_ranges = [(-100, 100)] * 30
    else:
        output_text.insert(tk.END, "Nieznana funkcja celu\n")
        return

    num_vars = len(variable_ranges)

    # Inicjalizacja populacji
    population = PopulationReal(num_vars, variable_ranges)
    population.create_initial_population(pop_size)
    population.evaluate(fitness_function)

    best_list = []
    avg_list = []
    std_list = []

    start = time.time()

    for epoch in range(epochs):
        population.individuals.sort(key=lambda x: x.fitness)
        elite = population.individuals[0].clone()

        crossover = CrossoverReal(population.individuals, crossover_prob)

        # wybór metody krzyżowania
        if crossover_method == "arithmetic":
            offspring = crossover.arithmetic_crossover()
        elif crossover_method == "linear":
            offspring = crossover.linear_crossover()
        elif crossover_method == "alpha_blend":
            offspring = crossover.alpha_blend_crossover(alpha=0.5)
        elif crossover_method == "alpha_beta_blend":
            offspring = crossover.alpha_beta_blend_crossover(alpha=0.5, beta=0.5)
        elif crossover_method == "average":
            offspring = crossover.average_crossover()
        else:
            output_text.insert(tk.END, "Nieznana metoda krzyżowania\n")
            return

        # wybór metody mutacji
        mutation = MutationReal(offspring, mutation_prob, variable_ranges)
        if mutation_method == "gaussian":
            mutated = mutation.gaussian_mutation(sigma=0.5)
        elif mutation_method == "uniform":
            mutated = mutation.uniform_mutation()
        else:
            output_text.insert(tk.END, "Nieznana metoda mutacji\n")
            return

        population.individuals = mutated
        population.individuals[0] = elite
        population.evaluate(fitness_function)

        fitnesses = [ind.fitness for ind in population.individuals]
        best = min(fitnesses)
        avg = sum(fitnesses) / len(fitnesses)
        std = (sum((x - avg) ** 2 for x in fitnesses) / len(fitnesses)) ** 0.5

        best_list.append(best)
        avg_list.append(avg)
        std_list.append(std)

        output_text.insert(tk.END, f"Epoka {epoch+1}: Best = {best:.6f}, Avg = {avg:.6f}, Std = {std:.6f}\n")
        output_text.see(tk.END)
        root.update()

    end = time.time()
    output_text.insert(tk.END, f"Czas wykonania: {end - start:.2f} s\n")

    # Zapis do pliku CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_real_{timestamp}.csv"
    with open(filename, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Best", "Average", "StdDev"])
        for i in range(epochs):
            writer.writerow([i + 1, best_list[i], avg_list[i], std_list[i]])

    # Wykres
    plt.figure()
    plt.plot(best_list, label="Best")
    plt.plot(avg_list, label="Average")
    plt.plot(std_list, label="StdDev")
    plt.xlabel("Epoka")
    plt.ylabel("Fitness")
    plt.title("Statystyki algorytmu genetycznego (Reprezentacja rzeczywista)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------- GUI --------------------
root = tk.Tk()
root.title("Algorytm genetyczny – Reprezentacja rzeczywista")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Parametry
tk.Label(frame, text="Liczba epok:").grid(row=0, column=0, sticky="e")
entry_epochs = tk.Entry(frame)
entry_epochs.grid(row=0, column=1)

tk.Label(frame, text="Rozmiar populacji:").grid(row=1, column=0, sticky="e")
entry_population = tk.Entry(frame)
entry_population.grid(row=1, column=1)

tk.Label(frame, text="Prawd. mutacji:").grid(row=2, column=0, sticky="e")
entry_mutation = tk.Entry(frame)
entry_mutation.grid(row=2, column=1)

tk.Label(frame, text="Prawd. krzyżowania:").grid(row=3, column=0, sticky="e")
entry_crossover = tk.Entry(frame)
entry_crossover.grid(row=3, column=1)

tk.Label(frame, text="Funkcja celu:").grid(row=4, column=0, sticky="e")
combo_function = ttk.Combobox(frame, values=["hypersphere", "cec_hybrid"])
combo_function.grid(row=4, column=1)

tk.Label(frame, text="Metoda krzyżowania:").grid(row=5, column=0, sticky="e")
combo_crossover = ttk.Combobox(frame, values=["arithmetic", "linear", "alpha_blend", "alpha_beta_blend", "average"])
combo_crossover.grid(row=5, column=1)

tk.Label(frame, text="Metoda mutacji:").grid(row=6, column=0, sticky="e")
combo_mutation = ttk.Combobox(frame, values=["gaussian", "uniform"])
combo_mutation.grid(row=6, column=1)

tk.Button(frame, text="Uruchom", command=run_algorithm).grid(row=7, column=0, columnspan=2, pady=10)

# Pole wyjściowe
output_text = tk.Text(root, height=20, width=80)
output_text.pack(padx=10, pady=10)

root.mainloop()
