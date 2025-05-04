import random

class ChromosomeReal:
    def __init__(self, num_of_variables, variables_ranges):
        """
        num_of_variables: liczba zmiennych
        variables_ranges: lista przedziałów [(min1, max1), (min2, max2), ...]
        """
        self.num_of_variables = num_of_variables
        self.variables_ranges = variables_ranges
        self.variables = []  # zmienne rzeczywiste
        self.fitness = None  # wartość funkcji celu

    def generate_chromosome(self):
        self.variables = [
            random.uniform(min_val, max_val)
            for (min_val, max_val) in self.variables_ranges
        ]

    def decode_variables(self):
        return self.variables

    def clone(self):
        clone = ChromosomeReal(self.num_of_variables, self.variables_ranges)
        clone.variables = self.variables.copy()
        clone.fitness = self.fitness
        return clone
