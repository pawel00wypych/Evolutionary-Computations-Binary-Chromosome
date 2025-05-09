import random

class MutationReal:
    def __init__(self, individuals, mutation_probability, variable_ranges):
        """
        individuals: lista osobników (ChromosomeReal)
        mutation_probability: prawdopodobieństwo mutacji (np. 0.1)
        variable_ranges: zakresy zmiennych [(min, max), (min, max), ...]
        """
        self.individuals = individuals
        self.mutation_probability = mutation_probability
        self.variable_ranges = variable_ranges

    def uniform_mutation(self):
        """Mutacja równomierna: losuje nową wartość zmiennej z zakresu"""
        for individual in self.individuals:
            for i in range(individual.num_of_variables):
                if random.random() < self.mutation_probability:
                    min_val, max_val = self.variable_ranges[i]
                    new_value = random.uniform(min_val, max_val)
                    individual.mutate_gene(i, new_value)
        return self.individuals

    def gaussian_mutation(self, sigma=1.0):
        """
        Mutacja Gaussa: dodaje do zmiennej szum z rozkładu normalnego
        Uwaga: wartości są przycinane do dozwolonego zakresu!
        """
        for individual in self.individuals:
            for i in range(individual.num_of_variables):
                if random.random() < self.mutation_probability:
                    current_value = individual.variables[i]
                    min_val, max_val = self.variable_ranges[i]
                    mutated_value = current_value + random.gauss(0, sigma)
                    # Przytnij do zakresu
                    mutated_value = max(min(mutated_value, max_val), min_val)
                    individual.mutate_gene(i, mutated_value)
        return self.individuals
