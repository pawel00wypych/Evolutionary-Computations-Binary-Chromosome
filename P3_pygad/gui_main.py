import tkinter as tk
from tkinter import ttk
import pygad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from custom_crossover import crossover_average_func, crossover_arithmetic_func, crossover_linear_func, crossover_alpha_func, crossover_alpha_beta_func
from custom_mutation import mutation_gaussian_func, mutation_uniform_func
from custom_crossover_binary import crossover_single_point, crossover_two_point
from custom_mutation_binary import mutation_single_point, mutation_two_point, mutation_edge
import benchmark_setup

class GeneticApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Genetic Algorithm GUI")
        self.geometry("550x600")
        self.create_widgets()

    def create_widgets(self):
        # Reprezentacja
        tk.Label(self, text="Reprezentacja:").pack()
        self.representation_var = tk.StringVar(value="real")
        ttk.Combobox(self, textvariable=self.representation_var, values=["real", "binary"]).pack()

        # Funkcja celu
        tk.Label(self, text="Funkcja celu:").pack()
        self.function_var = tk.StringVar(value="hyperellipsoid")
        ttk.Combobox(self, textvariable=self.function_var, values=["hyperellipsoid", "cec_f3"]).pack()

        # Krzyżowanie
        tk.Label(self, text="Krzyżowanie:").pack()
        self.crossover_var = tk.StringVar(value="average")
        ttk.Combobox(self, textvariable=self.crossover_var, values=["average", "arithmetic", "linear", "alpha", "alpha_beta", "single_point", "two_point"]).pack()

        # Mutacja
        tk.Label(self, text="Mutacja:").pack()
        self.mutation_var = tk.StringVar(value="gaussian")
        ttk.Combobox(self, textvariable=self.mutation_var, values=["gaussian", "uniform", "single_point", "two_point", "edge"]).pack()

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
        selected_function = self.function_var.get()
        selected_crossover = self.crossover_var.get()
        selected_mutation = self.mutation_var.get()

        # ustawienie funkcji benchmarkowej
        benchmark_setup.set_function(selected_function)
        self.evaluate_fitness = benchmark_setup.evaluate_fitness
        self.bounds = benchmark_setup.get_bounds()
        self.num_variables = benchmark_setup.get_dimensions()

        # rozpakowanie zakresu
        try:
            lower, upper = self.bounds[0]
        except Exception:
            lower, upper = -100, 100

        if representation == "real":
            crossover_map = {
                "average": crossover_average_func,
                "arithmetic": crossover_arithmetic_func,
                "linear": crossover_linear_func,
                "alpha": crossover_alpha_func,
                "alpha_beta": crossover_alpha_beta_func
            }
            mutation_map = {
                "gaussian": mutation_gaussian_func,
                "uniform": mutation_uniform_func
            }
            crossover_func = crossover_map.get(selected_crossover, crossover_average_func)
            mutation_func = mutation_map.get(selected_mutation, mutation_gaussian_func)
            self.run_real(generations, population, elitism, mutation_percent, crossover_func, mutation_func, lower, upper)
        else:
            crossover_map = {
                "single_point": crossover_single_point,
                "two_point": crossover_two_point
            }
            mutation_map = {
                "single_point": mutation_single_point,
                "two_point": mutation_two_point,
                "edge": mutation_edge
            }
            crossover_func = crossover_map.get(selected_crossover, crossover_two_point)
            mutation_func = mutation_map.get(selected_mutation, mutation_edge)
            self.run_binary(generations, population, elitism, mutation_percent, crossover_func, mutation_func, lower, upper)

    def run_real(self, generations, population, elitism, mutation_percent, crossover_func, mutation_func, lower, upper):
        bests, means, stds = [], [], []

        def fitness_func(ga, individual, _):
            return self.evaluate_fitness(individual)

        def on_gen(ga):
            fitnesses = -np.array(ga.last_generation_fitness)
            bests.append(fitnesses.min())
            means.append(fitnesses.mean())
            stds.append(fitnesses.std())

        ga = pygad.GA(
            num_generations=generations,
            sol_per_pop=population,
            num_parents_mating=population//2,
            num_genes=self.num_variables,
            init_range_low=lower,
            init_range_high=upper,
            gene_type=float,
            fitness_func=fitness_func,
            crossover_type=crossover_func,
            mutation_type=mutation_func,
            mutation_percent_genes=mutation_percent,
            keep_elitism=elitism,
            on_generation=on_gen
        )
        ga.run()
        self.plot_results(bests, means, stds)
        self.save_results(bests, means, stds)

    def run_binary(self, generations, population, elitism, mutation_percent, crossover_func, mutation_func, lower, upper):
        bits_per_variable = 20
        num_genes = self.num_variables * bits_per_variable
        bests, means, stds = [], [], []

        def decode(ind):
            out = []
            for i in range(self.num_variables):
                b = ''.join(str(bit) for bit in ind[i*bits_per_variable:(i+1)*bits_per_variable])
                dec = int(b, 2)
                real = lower + (dec / (2**bits_per_variable - 1)) * (upper - lower)
                out.append(real)
            return out

        def fitness_func(ga, individual, _):
            return self.evaluate_fitness(decode(individual))

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
            crossover_type=crossover_func,
            mutation_type=mutation_func,
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