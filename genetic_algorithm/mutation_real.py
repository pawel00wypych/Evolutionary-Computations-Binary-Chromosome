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
                    low, high = individual.variables_ranges_list[i]
                    new_value = random.uniform(low, high)
                    # przycinanie do zakresu
                    new_value = max(min(new_value, high), low)
                    individual.mutate_gene(i, new_value)
        return self.individuals

    def gaussian_mutation(self, sigma=1.0):
        # mutacja Gaussa: dodaje do zmiennej szum z rozkładu normalnego
        # Uwaga: wartości są przycinane do dozwolonego zakresu!
        for individual in self.individuals:
            for i in range(individual.num_of_variables):
                if random.random() < self.mutation_probability:
                    # bieżąca wartość zmiennej
                    value = individual.variables[i]
                    low, high = individual.variables_ranges_list[i]
                    stddev = (high - low) * 0.1
                    mutated = value + random.gauss(0, stddev)
                    # przycinanie do zakresu
                    individual.variables[i] = min(max(mutated, low), high)
        return self.individuals
