import random

class ChromosomeReal:
    def __init__(self, num_of_variables, variables_ranges):
        """
        num_of_variables: ile zmiennych (np. 3 dla x, y, z)
        variables_ranges: lista krotek [(min1, max1), (min2, max2), ...]
        """
        self.num_of_variables = num_of_variables
        self.variables_ranges = variables_ranges  # [(min, max), (min, max), ...]
        self.variables = []  # lista zmiennych rzeczywistych
        self.fitness = None  # wartość przystosowania (fitness)

    def generate_chromosome(self):
        """Losowo generuje zmienne rzeczywiste w zadanych przedziałach"""
        self.variables = [
            random.uniform(min_val, max_val)
            for (min_val, max_val) in self.variables_ranges
        ]

    def decode_variables(self):
        """Zwraca zmienne rzeczywiste"""
        return self.variables

    def mutate_gene(self, gene_idx, new_value):
        """Modyfikacja jednej zmiennej na nową wartość (do mutacji rzeczywistej)"""
        if 0 <= gene_idx < len(self.variables):
            self.variables[gene_idx] = new_value

    def clone(self):
        """Tworzy głęboką kopię chromosomu"""
        clone = ChromosomeReal(self.num_of_variables, self.variables_ranges)
        clone.variables = self.variables.copy()
        clone.fitness = self.fitness
        return clone

