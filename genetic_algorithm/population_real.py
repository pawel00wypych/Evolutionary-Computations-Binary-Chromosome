from genetic_algorithm.chromosome_real import ChromosomeReal

class PopulationReal:
    def __init__(self, num_variables, variable_ranges):
        """
        num_variables: liczba zmiennych (np. 3)
        variable_ranges: lista przedziałów [(min, max), (min, max), ...]
        """
        self.num_variables = num_variables
        self.variable_ranges = variable_ranges
        self.individuals = []

    def create_initial_population(self, size):
        self.individuals = []
        for _ in range(size):
            chromo = ChromosomeReal(self.num_variables, self.variable_ranges)
            chromo.generate_chromosome()
            self.individuals.append(chromo)

    def evaluate(self, fitness_function):
        for individual in self.individuals:
            individual.fitness = fitness_function(individual.decode_variables())
