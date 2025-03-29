import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib import pyplot as plt
from genetic_algorithm import evaluation_functions
from .main import run_genetic_algorithm
import numpy as np

class GeneticApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Genetic Algorithm GUI")
        self.geometry("500x620")
        self.create_widgets()

    def create_widgets(self):
        # --- FUNCTION SELECTION ---
        ttk.Label(self, text="Function:").pack(pady=(10, 0))
        self.function_var = tk.StringVar()
        self.function_box = ttk.Combobox(self, textvariable=self.function_var, state="readonly")
        self.function_box["values"] = ["Hypersphere", "Hybrid CEC 2014 (F1)", "Rana", "Composition 6"]
        self.function_box.current(0)
        self.function_box.pack()

        # --- BASIC CONFIG ---
        def field(label):
            ttk.Label(self, text=label).pack()
            entry = ttk.Entry(self)
            entry.pack()
            return entry

        self.stop_criteria_var = field("Num of epochs without change to stop:")
        self.variables_entry = field("Number of variables:")
        self.precision_entry = field("Precision (e.g. 6):")
        self.population_entry = field("Population size:")
        self.epochs_entry = field("Epochs:")
        self.crossover_prob_entry = field("Crossover probability (0-1):")
        self.mutation_prob_entry = field("Mutation probability (0-1):")
        self.inversion_prob_entry = field("Inversion probability (0-1):")
        self.inversion_level_entry = field("inversion_level (0-1):")

        # --- SELECTION METHODS ---
        def combo(label, values):
            ttk.Label(self, text=label).pack()
            var = tk.StringVar()
            box = ttk.Combobox(self, textvariable=var, state="readonly")
            box["values"] = values
            box.current(0)
            box.pack()
            return var

        self.selection_var = combo("Selection method:", ["tournament", "roulette", "best"])
        self.selection_type_var = combo("Selection method:", ["min", "max"])
        self.crossover_var = combo("Crossover method:", ["single", "two", "granular", "uniform"])
        self.mutation_var = combo("Mutation method:", ["single", "two", "edge"])

        # --- START BUTTON ---
        ttk.Button(self, text="Start", command=self.run_algorithm).pack(pady=10)

        # --- RESULT DISPLAY ---
        self.result_text = tk.Text(self, height=6, wrap="word")
        self.result_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def run_algorithm(self):
        try:
            # --- Extract GUI values ---
            fun_name = self.function_var.get()
            num_vars = int(self.variables_entry.get())
            precision = int(self.precision_entry.get())
            population = int(self.population_entry.get())
            epochs = int(self.epochs_entry.get())
            crossover_p = float(self.crossover_prob_entry.get())
            mutation_p = float(self.mutation_prob_entry.get())
            inversion_p = float(self.inversion_prob_entry.get())
            inversion_level = float(self.inversion_level_entry.get())
            selection = self.selection_var.get()
            selection_type = self.selection_type_var.get()
            crossover = self.crossover_var.get()
            mutation = self.mutation_var.get()
            stop_criteria = int(self.stop_criteria_var.get())

            # --- Function setup ---
            if fun_name == "Hypersphere":
                fitness = evaluation_functions.hypersphere_fitness
                if selection_type == "min":
                    expected = evaluation_functions.get_hypersphere_minimum()
                else:
                    expected = evaluation_functions.get_hypersphere_maximum()
                ranges = [(-5, 5)]
            elif fun_name == "Hybrid CEC 2014 (F1)":
                fitness = evaluation_functions.hybrid_fitness
                expected = evaluation_functions.get_cec_hybrid_minimum()
                ranges = [(-100, 100)]
            elif fun_name == "Rana":
                fitness = evaluation_functions.rana_fitness
                if selection_type == "min":
                    expected = evaluation_functions.get_rana_minimum()
                else:
                    expected = evaluation_functions.get_rana_maximum()
                ranges = [(-512, 512)]
            elif fun_name == "Composition 6":
                fitness = evaluation_functions.composition_6_fitness
                expected = evaluation_functions.get_cec_composition_6_minimum()
                ranges = [(-100, 100)]

            # --- Create config dict ---
            config = {
                "fitness_function": fitness,
                "expected_minimum": expected,
                "num_of_variables": num_vars,
                "variables_ranges_list": ranges,
                "precision": precision,
                "mutation_probability": mutation_p,
                "crossover_probability": crossover_p,
                "inversion_probability": inversion_p,
                "inversion_level": inversion_level,
                "selection_method": selection,
                "selection_type": selection_type,
                "crossover_method": crossover,
                "mutation_method": mutation,
                "epochs": epochs,
                "population_size": population,
                "stop_criteria": stop_criteria
            }

            # --- Run algorithm ---
            results = []
            execution_times = []
            best_solution = None
            best_fitness = float('inf')
            best_history = None

            for i in range(10):
                result = run_genetic_algorithm(config)
                results.append(result['best_fitness'])
                execution_times.append(result['execution_time'])

                # Keep track of the best solution
                if result['best_fitness'] < best_fitness:
                    best_fitness = result['best_fitness']
                    best_solution = result['best_solution']
                    best_history = result['history']

            # --- Calculate statistics ---
            average_result = np.mean(results)
            average_time = np.mean(execution_times)
            best_result = np.min(results)
            worst_result = np.max(results)

            # --- Show result ---
            text = f"Average fitness: {average_result:.6f}\n"
            text += f"Best fitness: {best_result:.6f}\n"
            text += f"Worst fitness: {worst_result:.6f}\n"
            text += f"Average execution time: {average_time:.2f}s\n"
            text += f"Best solution: {best_solution}\n"
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, text)

            # --- Show plots for best result ---
            self.plot_results(best_history)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_results(self, history):
        plt.figure()
        plt.plot(history["best_fitness"], label="Best value")
        plt.title("Best fitness")
        plt.xlabel("Epoch")
        plt.ylabel("Value of function")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(history["avg_fitness"], label="average")
        plt.plot(history["std_fitness"], label="standard deviation")
        plt.title("Average and standard deviation of the function")
        plt.xlabel("Epoch")
        plt.ylabel("Value of function")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    app = GeneticApp()
    app.mainloop()
