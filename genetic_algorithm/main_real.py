import time
from statistics import mean, stdev
from genetic_algorithm.population_real import PopulationReal
from genetic_algorithm.crossover_real import CrossoverReal
from genetic_algorithm.mutation_real import MutationReal
from genetic_algorithm.elitism import Elitism
from genetic_algorithm.selection import Selection

def run_real_genetic_algorithm(config):
    fitness_function = config["fitness_function"]
    num_of_variables = config["num_of_variables"]
    mutation_probability = config["mutation_probability"]
    crossover_probability = config["crossover_probability"]
    expected_minimum = config["expected_minimum"]
    selection_method = config["selection_method"]
    selection_type = config["selection_type"]
    crossover_method = config["crossover_method"]
    mutation_method = config["mutation_method"]
    epochs = config["epochs"]
    population_size = config["population_size"]
    stop_criteria = config["stop_criteria"]

    history = {
        "best_fitness": [],
        "avg_fitness": [],
        "std_fitness": [],
    }

    start_time = time.time()
    population = PopulationReal(num_of_variables, population_size)
    population.evaluate(fitness_function)

    selection_map = {
        "tournament": Selection.tournament_selection,
        "roulette": Selection.roulette_selection,
        "best": Selection.best_selection
    }

    crossover_map = {
        "arithmetic": CrossoverReal.arithmetic_crossover,
        "linear": CrossoverReal.linear_crossover,
        "alpha": CrossoverReal.alpha_blend_crossover,
        "alpha_beta": CrossoverReal.alpha_beta_blend_crossover,
        "average": CrossoverReal.average_crossover
    }

    mutation_map = {
        "uniform": MutationReal.uniform_mutation,
        "gaussian": MutationReal.gaussian_mutation
    }

    best_fitness = float("inf") if selection_type == "min" else float("-inf")
    no_improvement_counter = 0

    for epoch in range(epochs):
        elitism_operator = Elitism(population.individuals)
        elitism_operator.choose_the_best_individuals()
        elites = elitism_operator.get_elite_list()

        crossover_operator = CrossoverReal(population.individuals, crossover_probability, elitism_operator.number_of_elites)
        offspring = crossover_map[crossover_method](crossover_operator)

        mutation_operator = MutationReal(offspring, mutation_probability)
        mutated_offspring = mutation_map[mutation_method](mutation_operator)

        new_population = elites + mutated_offspring
        population.individuals = new_population
        population.evaluate(fitness_function)

        selected = selection_map[selection_method](population, selection_type=selection_type, num_selected=population_size)
        new_best_fitness = selected[0].fitness

        if (selection_type == "min" and new_best_fitness < best_fitness) or \
           (selection_type == "max" and new_best_fitness > best_fitness):
            best_fitness = new_best_fitness
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        fitness_values = [ind.fitness for ind in population.individuals]
        avg_fitness = mean(fitness_values)
        std_dev = stdev(fitness_values) if len(fitness_values) > 1 else 0

        history["best_fitness"].append(best_fitness)
        history["avg_fitness"].append(avg_fitness)
        history["std_fitness"].append(std_dev)

        if no_improvement_counter >= stop_criteria:
            print(f"Algorithm stopped – no improvement for {stop_criteria} epochs")
            break

    end_time = time.time()
    execution_time = end_time - start_time

    return {
        "best_solution": selected[0].decoded_variables,
        "best_fitness": best_fitness,
        "expected_minimum": expected_minimum,
        "execution_time": execution_time,
        "history": history
    }