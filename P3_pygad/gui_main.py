import tkinter as tk
from tkinter import ttk
import pygad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from benchmark_setup import evaluate_fitness, get_bounds, get_dimensions
from custom_crossover import crossover_average_func
from custom_mutation import mutation_gaussian_func
from custom_crossover_binary import crossover_two_point
from custom_mutation_binary import mutation_edge

class GeneticApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Genetic Algorithm GUI")
        self.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        # Reprezentacja
        tk.Label(self, text="Reprezentacja:").pack()
        self.representation_var = tk.StringVar(value="real")
        ttk.Combobox(self, textvariable=self.representation_var, values=["real", "binary"]).pack()

        # Generacje
        tk.Label(self, text="Liczba generacji:").pack()
        self.gen_entry = tk.Entry(self)
        self.gen_entry.insert(0, "100")
        self.gen_entry.pack()

        # Populacja
        tk.Label(self, text="Rozmiar populacji:").pack()
        self.pop_entry = tk.Entry(self)
        self.pop_entry.insert(0, "50")
        self.pop_entry.pack()

        # Elityzm
        tk.Label(self, text="Elityzm:").pack()
        self.elite_entry = tk.Entry(self)
        self.elite_entry.insert(0, "2")
        self.elite_entry.pack()

        # Mutacja
        tk.Label(self, text="Mutacja (%):").pack()
        self.mut_entry = tk.Entry(self)
        self.mut_entry.insert(0, "5")
        self.mut_entry.pack()

        # Start
        tk.Button(self, text="Start", command=self.run_ga).pack(pady=10)

    def run_ga(self):
        generations = int(self.gen_entry.get())
        population = int(self.pop_entry.get())
        elitism = int(self.elite_entry.get())
        mutation_percent = int(self.mut_entry.get())
        representation = self.representation_var.get()

        if representation == "real":
            self.run_real(generations, population, elitism, mutation_percent)
        else:
            self.run_binary(generations, population, elitism, mutation_percent)

    def run_real(self, generations, population, elitism, mutation_percent):
        num_variables = get_dimensions()
        bounds = get_bounds()
        lower, upper = bounds[0]
        bests, means, stds = [], [], []

        def fitness_func(ga, individual, _):
            return evaluate_fitness(individual)

        def on_gen(ga):
            fitnesses = -np.array(ga.last_generation_fitness)
            bests.append(fitnesses.min())
            means.append(fitnesses.mean())
            stds.append(fitnesses.std())

        ga = pygad.GA(
            num_generations=generations,
            sol_per_pop=population,
            num_parents_mating=population//2,
            num_genes=num_variables,
            init_range_low=lower,
            init_range_high=upper,
            gene_type=float,
            fitness_func=fitness_func,
            crossover_type=crossover_average_func,
            mutation_type=mutation_gaussian_func,
            mutation_percent_genes=mutation_percent,
            keep_elitism=elitism,
            on_generation=on_gen
        )
        ga.run()
        self.plot_results(bests, means, stds)
        self.save_results(bests, means, stds)

    def run_binary(self, generations, population, elitism, mutation_percent):
        num_variables = get_dimensions()
        bits_per_variable = 20
        num_genes = num_variables * bits_per_variable
        bounds = get_bounds()
        lower, upper = bounds[0]
        bests, means, stds = [], [], []

        def decode(ind):
            out = []
            for i in range(num_variables):
                b = ''.join(str(bit) for bit in ind[i*bits_per_variable:(i+1)*bits_per_variable])
                dec = int(b, 2)
                real = lower + (dec / (2**bits_per_variable - 1)) * (upper - lower)
                out.append(real)
            return out

        def fitness_func(ga, individual, _):
            return evaluate_fitness(decode(individual))

        def on_gen(ga):
            fitnesses = -np.array(ga.last_generation_fitness)
            bests.append(fitnesses.min())
            means.append(fitnesses.mean())
            stds.append(fitnesses.std())

        ga = pygad.GA(
            num_generations=generations,
            sol_per_pop=population,
            num_parents_mating=population//2,
            num_genes=num_genes,
            init_range_low=0,
            init_range_high=2,
            gene_type=int,
            fitness_func=fitness_func,
            crossover_type=crossover_two_point,
            mutation_type=mutation_edge,
            mutation_percent_genes=mutation_percent,
            keep_elitism=elitism,
            on_generation=on_gen
        )
        ga.run()
        self.plot_results(bests, means, stds)
        self.save_results(bests, means, stds)

    def plot_results(self, bests, means, stds):
        plt.plot(bests, label="Najlepszy")
        plt.plot(means, label="Średnia")
        plt.plot(stds, label="Odchylenie")
        plt.xlabel("Generacja")
        plt.ylabel("Wartość funkcji celu")
        plt.title("Postęp optymalizacji")
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_results(self, bests, means, stds):
        df = pd.DataFrame({
            "Generacja": list(range(1, len(bests)+1)),
            "Najlepszy": bests,
            "Średnia": means,
            "Odchylenie": stds
        })
        df.to_csv("wyniki_ga.csv", index=False)
        print("Wyniki zapisane do wyniki_ga.csv")

if __name__ == "__main__":
    app = GeneticApp()
    app.mainloop()
