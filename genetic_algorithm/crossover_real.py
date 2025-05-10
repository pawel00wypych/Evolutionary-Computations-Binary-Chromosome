import random
from genetic_algorithm.chromosome_real import ChromosomeReal


class CrossoverReal:
    def __init__(self, parents, crossover_probability, num_of_elite_individuals, variable_ranges_list):
        self.parents = parents
        self.crossover_probability = crossover_probability
        self.num_of_elite_individuals = num_of_elite_individuals
        self.variable_ranges = variable_ranges_list
        self.offspring = []

    def _prepare_pairs(self):
        # losowe parowanie osobników do krzyżowania
        parents = self.parents[:]
        random.shuffle(parents)
        return [(parents[i], parents[i+1]) for i in range(0, len(parents) - 1, 2)]

    def _clip(self, value, i):
        min_val, max_val = self.variable_ranges[i]
        return max(min(value, max_val), min_val)
    
    def arithmetic_crossover(self):
        # krzyżowanie arytmetyczne: średnia z wartości rodziców
        self.offspring = []
        elites = sorted(self.parents, key=lambda x: x.fitness, reverse=True)[:self.num_of_elite_individuals]
        for p1, p2 in self._prepare_pairs():
            if random.random() <= self.crossover_probability:
                child1 = p1.clone()
                child2 = p2.clone()
                for i in range(p1.num_of_variables):
                    avg = (p1.variables[i] + p2.variables[i]) / 2
                    clipped = self._clip(avg, i)
                    child1.variables[i] = clipped
                    child2.variables[i] = clipped
                self.offspring.extend([child1, child2])
            else:
                self.offspring.extend([p1.clone(), p2.clone()])

        self.offspring.extend(elites)
        
        return self.offspring[:len(self.parents)]

    def linear_crossover(self):
        # krzyżowanie liniowe:
        # tworzy 3 dzieci: (c1 = 0.5*p1 + 0.5*p2, c2 = 1.5*p1 - 0.5*p2, c3 = -0.5*p1 + 1.5*p2)
        # wybieramy 2 najlepsze losowo
        
        self.offspring = []
        elites = sorted(self.parents, key=lambda x: x.fitness, reverse=True)[:self.num_of_elite_individuals]
        for p1, p2 in self._prepare_pairs():
            if random.random() <= self.crossover_probability:
                c1 = p1.clone()
                c2 = p1.clone()
                c3 = p1.clone()
                c1.variables = []
                c2.variables = []
                c3.variables = []
                for i in range(p1.num_of_variables):
                    x = p1.variables[i]
                    y = p2.variables[i]
                    d1 = self._clip(0.5 * x + 0.5 * y, i)
                    d2 = self._clip(1.5 * x - 0.5 * y, i)
                    d3 = self._clip(-0.5 * x + 1.5 * y, i)
                    c1.variables.append(d1)
                    c2.variables.append(d2)
                    c3.variables.append(d3)
                # Losowo wybierz 2 dzieci z 3
                self.offspring.extend(random.sample([c1, c2, c3], 2))
            else:
                self.offspring.extend([p1.clone(), p2.clone()])
    
        self.offspring.extend(elites)

        return self.offspring[:len(self.parents)]

    def alpha_blend_crossover(self, alpha=0.5):
        # krzyżowanie mieszające typu alfa:
        # dla każdego genu: dziecko = p1 + alpha * (p2 - p1)
        
        self.offspring = []
        elites = sorted(self.parents, key=lambda x: x.fitness, reverse=True)[:self.num_of_elite_individuals]
        for p1, p2 in self._prepare_pairs():
            if random.random() <= self.crossover_probability:
                c1 = p1.clone()
                c2 = p2.clone()
                for i in range(p1.num_of_variables):
                    x = p1.variables[i]
                    y = p2.variables[i]
                    c1.variables[i] = self._clip(x + alpha * (y - x), i)
                    c2.variables[i] = self._clip(y + alpha * (x - y), i)
                self.offspring.extend([c1, c2])
            else:
                self.offspring.extend([p1.clone(), p2.clone()])

        self.offspring.extend(elites)

        return self.offspring[:len(self.parents)]

    def alpha_beta_blend_crossover(self, alpha=0.5, beta=0.5):
        # krzyżowanie mieszające typu alfa-beta:
        # dziecko = losowa wartość z przedziału: [min(x, y) - alpha * d, max(x, y) + beta * d] gdzie d = |x - y|
        
        self.offspring = []
        elites = sorted(self.parents, key=lambda x: x.fitness, reverse=True)[:self.num_of_elite_individuals]
        for p1, p2 in self._prepare_pairs():
            if random.random() <= self.crossover_probability:
                c1 = p1.clone()
                c2 = p2.clone()
                for i in range(p1.num_of_variables):
                    x = p1.variables[i]
                    y = p2.variables[i]
                    d = abs(x - y)
                    low = min(x, y) - alpha * d
                    high = max(x, y) + beta * d
                    c1.variables[i] = self._clip(random.uniform(low, high), i)
                    c2.variables[i] = self._clip(random.uniform(low, high), i)
                self.offspring.extend([c1, c2])
            else:
                self.offspring.extend([p1.clone(), p2.clone()])

        self.offspring.extend(elites)

        return self.offspring[:len(self.parents)]

    def average_crossover(self):
        # krzyżowanie uśredniające:
        # dla każdego genu: dziecko = (p1 + p2) / 2
        # (to samo co arytmetyczne, powtórka tylko dla formalności)
        return self.arithmetic_crossover()
