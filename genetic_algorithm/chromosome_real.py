import random

class ChromosomeReal:
    def __init__(self, num_of_variables, precision, variables_ranges_list, fitness=0):
        # definiuje ilość zmiennych i krotek
        self.num_of_variables = num_of_variables
        self.variables_ranges_list = variables_ranges_list  # [(min, max), (min, max), ...]
        self.variables = []  # lista zmiennych rzeczywistych
        self.fitness = fitness  # wartość przystosowania (fitness)
        self.precision = precision

    def random_in_range(self, min_val, max_val):
            value = random.uniform(min_val, max_val)
            if self.precision is not None:
                return round(value, self.precision)  # zastosowanie precyzji
            return value

    def generate_chromosome(self):
        # losowo generuje zmienne w zadanych przedziałach
        self.variables = [
            random.uniform(min_val, max_val)
            for (min_val, max_val) in self.variables_ranges_list
        ]

    def decode_variables(self):
        # zwraca zmienne rzeczywiste
        return self.variables

    def mutate_gene(self, gene_idx, mutation_range):
        # modyfikacja jednej zmiennej
        if 0 <= gene_idx < len(self.variables):
            min_val, max_val = self.variables_ranges_list[gene_idx]
            mutation_value = random.uniform(min_val - mutation_range, max_val + mutation_range)
            if self.precision is not None:
                mutation_value = round(mutation_value, self.precision)  # zastosowanie precyzji
            self.variables[gene_idx] = mutation_value

    def clone(self):
        # kopia chromosomu
        clone = ChromosomeReal(self.num_of_variables, self.precision, self.variables_ranges_list, self.fitness)
        clone.variables = self.variables.copy()
        clone.fitness = self.fitness
        return clone
    
    def evaluate_fitness(self, fitness_function):
        # wartość przystosowania (fitness) na podstawie funkcji przystosowania
        if self.fitness is None:
            self.fitness = fitness_function(self.decode_variables())
        return self.fitness

    def __repr__(self):
        return f"ChromosomeReal(variables={self.variables}, fitness={self.fitness})"

