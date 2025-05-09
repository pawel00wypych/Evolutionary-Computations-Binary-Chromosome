import random

class MutationReal:
    def __init__(self, individuals, mutation_probability):
        self.individuals = individuals
        self.mutation_probability = mutation_probability

    def uniform_mutation(self):
        # mutacja równomierna: losuje nową wartość zmiennej z zakresu
        # Uwaga: wartości są przycinane do dozwolonego zakresu!
        for individual in self.individuals:
            for i in range(individual.num_of_variables):
                if random.random() < self.mutation_probability:
                    # wybieramy nową wartość zmiennej z zakresu
                    min_val, max_val = individual.variables_ranges[i]
                    new_value = random.uniform(min_val, max_val)
                    # przycinanie do zakresu
                    new_value = max(min(new_value, max_val), min_val)
                    individual.mutate_gene(i, new_value)
        return self.individuals

    def gaussian_mutation(self, sigma=1.0):
        # mutacja Gaussa: dodaje do zmiennej szum z rozkładu normalnego
        # Uwaga: wartości są przycinane do dozwolonego zakresu!
        for individual in self.individuals:
            for i in range(individual.num_of_variables):
                if random.random() < self.mutation_probability:
                    # bieżąca wartość zmiennej
                    current_value = individual.variables[i]
                    min_val, max_val = individual.variables_ranges[i]
                    # szum
                    mutated_value = current_value + random.gauss(0, sigma)
                    # przycinanie do zakresu
                    mutated_value = max(min(mutated_value, max_val), min_val)
                    individual.mutate_gene(i, mutated_value)
        return self.individuals
