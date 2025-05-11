import tkinter as tk
from tkinter import ttk
import pygad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

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
        representation_combo = ttk.Combobox(self, textvariable=self.representation_var, values=["real", "binary"])
        representation_combo.pack()
        representation_combo.bind("<<ComboboxSelected>>", self.update_options)

        # Funkcja celu
        tk.Label(self, text="Funkcja celu:").pack()
        self.function_var = tk.StringVar(value="hypersphere")
        ttk.Combobox(self, textvariable=self.function_var, values=["hypersphere", "rana", "hyperellipsoid", "Hybrid CEC 2014 (F1)", "Composition 6", "cec_f3"]).pack()
        

        # Krzyżowanie
        tk.Label(self, text="Krzyżowanie:").pack()
        self.crossover_var = tk.StringVar()
        self.crossover_combobox = ttk.Combobox(self, textvariable=self.crossover_var)
        self.crossover_combobox.pack()

        # Mutacja
        tk.Label(self, text="Mutacja:").pack()
        self.mutation_var = tk.StringVar()
        self.mutation_combobox = ttk.Combobox(self, textvariable=self.mutation_var)
        self.mutation_combobox.pack()

        # Generacje
        tk.Label(self, text="Liczba generacji:").pack()
        self.gen_entry = tk.Entry(self)
        self.gen_entry.insert(0, "100")
        self.gen_entry.pack()

        # Populacja
        tk.Label(self, text="Rozmiar populacji:").pack()
        self.pop_entry = tk.Entry(self)
        self.pop_entry.insert(0, "100")
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

        # Krzyżowanie
        tk.Label(self, text="Krzyżowanie (%):").pack()
        self.cross_entry = tk.Entry(self)
        self.cross_entry.insert(0, "5")
        self.cross_entry.pack()

        # Start
        tk.Button(self, text="Start", command=self.run_ga).pack(pady=10)
        
        self.update_options()

    def update_options(self, *_):
        # dostępne opcje w zależności od reprezentacji
        representation = self.representation_var.get()

        if representation == "real":
            self.crossover_combobox["values"] = ["average", "arithmetic", "linear", "alpha", "alpha_beta"]
            self.crossover_var.set("average")

            self.mutation_combobox["values"] = ["gaussian", "uniform"]
            self.mutation_var.set("gaussian")
        else:  # binary
            self.crossover_combobox["values"] = ["single_point", "two_point"]
            self.crossover_var.set("two_point")

            self.mutation_combobox["values"] = ["single_point", "two_point", "edge"]
            self.mutation_var.set("edge")

    def run_ga(self):
        generations = int(self.gen_entry.get())
        population = int(self.pop_entry.get())
        elitism = int(self.elite_entry.get())
        mutation_percent = int(self.mut_entry.get())
        crossover_percent = int(self.cross_entry.get())
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
            self.run_real(generations, population, elitism, mutation_percent, crossover_percent, crossover_func, mutation_func, lower, upper)
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
            self.run_binary(generations, population, elitism, mutation_percent, crossover_percent, crossover_func, mutation_func, lower, upper)

    def run_real(self, generations, population, elitism, mutation_percent, crossover_percent, crossover_func, mutation_func, lower, upper):
        all_bests, all_means, all_stds = [], [], []
        fitness_scores, execution_times, best_solutions = [], [], []

        for i in range(5):
            print(f"Run {i+1}/5...")
            bests, means, stds = [], [], []

            def fitness_func(ga, individual, _):
                return self.evaluate_fitness(individual)

            def on_gen(ga):
                fitnesses = -np.array(ga.last_generation_fitness)
                bests.append(fitnesses.min())
                means.append(fitnesses.mean())
                stds.append(fitnesses.std())

            start_time = time.time()

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
                crossover_probability=crossover_percent/100,
                keep_elitism=elitism,
                on_generation=on_gen
            )

            ga.run()

            end_time = time.time() 
            execution_times.append(end_time - start_time)

            fitness_scores.append(-ga.best_solution()[1])
            best_solutions.append(ga.best_solution()[0])

            all_bests.append(bests)
            all_means.append(means)
            all_stds.append(stds)

        best_result = min(fitness_scores)
        worst_result = max(fitness_scores)
        average_result = sum(fitness_scores) / len(fitness_scores)
        average_time = sum(execution_times) / len(execution_times)
        best_index = fitness_scores.index(best_result)
        best_solution = best_solutions[best_index]

        # Wyświetlenie wyników
        text = f"Average fitness: {average_result:.6f}\n"
        text += f"Best fitness: {best_result:.6f}\n"
        text += f"Worst fitness: {worst_result:.6f}\n"
        text += f"Average execution time: {average_time:.2f}s\n"
        text += f"Best solution: {best_solution}\n"
        print(text)

        # Dwa wykresy z najlepszego przebiegu
        self.plot_results(all_bests[best_index], all_means[best_index], all_stds[best_index])
        self.save_results(all_bests[best_index], all_means[best_index], all_stds[best_index])

    def run_binary(self, generations, population, elitism, mutation_percent, crossover_percent, crossover_func, mutation_func, lower, upper):
        bits_per_variable = 20
        num_genes = self.num_variables * bits_per_variable
        all_bests, all_means, all_stds = [], [], []
        fitness_scores, execution_times, best_solutions = [], [], []

        for i in range(5):
            print(f"Run {i+1}/5...")
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

            start_time = time.time()

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
                crossover_probability=crossover_percent/100,
                keep_elitism=elitism,
                on_generation=on_gen
            )

            ga.run()

            end_time = time.time() 
            execution_times.append(end_time - start_time)

            fitness_scores.append(-ga.best_solution()[1])
            best_solutions.append(ga.best_solution()[0])

            all_bests.append(bests)
            all_means.append(means)
            all_stds.append(stds)

        best_result = min(fitness_scores)
        worst_result = max(fitness_scores)
        average_result = sum(fitness_scores) / len(fitness_scores)
        average_time = sum(execution_times) / len(execution_times)
        best_index = fitness_scores.index(best_result)
        best_solution = best_solutions[best_index]

        # Wyświetlenie wyników
        text = f"Average fitness: {average_result:.6f}\n"
        text += f"Best fitness: {best_result:.6f}\n"
        text += f"Worst fitness: {worst_result:.6f}\n"
        text += f"Average execution time: {average_time:.2f}s\n"
        text += f"Best solution: {best_solution}\n"
        print(text)

        # Dwa wykresy z najlepszego przebiegu
        self.plot_results(all_bests[best_index], all_means[best_index], all_stds[best_index])
        self.save_results(all_bests[best_index], all_means[best_index], all_stds[best_index])

    def plot_results(self, bests, means, stds):
        generations = list(range(1, len(bests)+1))

        # wykres 1 - fitness
        plt.figure(figsize=(10, 5))
        plt.plot(generations, bests, label="Najlepszy", color='blue')
        plt.title("Najlepszy wynik w każdej epoce")
        plt.xlabel("Epoka")
        plt.ylabel("Wartość funkcji celu")
        plt.grid(True)
        plt.legend()
        plt.show()

        # wykres 2 - średnia i odchylenie
        plt.figure(figsize=(10, 5))
        plt.plot(generations, means, label="Średnia", color='green')
        plt.plot(generations, stds, label="Odchylenie", color='orange')
        plt.title("Średnia i odchylenie standardowe")
        plt.xlabel("Epoka")
        plt.ylabel("Wartość funkcji celu")
        plt.grid(True)
        plt.legend()
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