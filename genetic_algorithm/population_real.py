from genetic_algorithm.chromosome_real import ChromosomeReal

class PopulationReal:
    def __init__(self, num_variables, precision, variables_ranges_list, population_size, individuals=None):
        self.size = population_size
        self.num_variables = num_variables
        self.variables_ranges_list = variables_ranges_list
        self.precision = precision
        if individuals is None:
            self.individuals = []
        else:
            self.individuals = individuals
        self.initialize()

        print("RANGES:", self.variables_ranges_list)
        print("NUM VARIABLES:", self.num_variables)
        
    def initialize(self):
        # inicjuje populację losowymi osobnikami
        self.individuals = [
            ChromosomeReal(self.num_variables, self.precision, self.variables_ranges_list)
            for _ in range(self.size)
        ]
        
        for individual in self.individuals:
            individual.generate_chromosome()

    def evaluate(self, fitness_function):
        # ocena populacji na podstawie funkcji fitness
        for individual in self.individuals:
            for individual in self.individuals:
                individual.fitness = fitness_function(individual.decode_variables())

    def get_population_size(self):
        # zwraca rozmiar populacji
        return len(self.individuals)
