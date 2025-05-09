import random
from genetic_algorithm.chromosome_real import ChromosomeReal


class CrossoverReal:
    def __init__(self, parents, crossover_probability, num_of_elite_individuals):
        self.parents = parents
        self.crossover_probability = crossover_probability
        self.num_of_elite_individuals = num_of_elite_individuals
        self.offspring = []

    def _prepare_pairs(self):
        # losowe parowanie osobników do krzyżowania
        parents = self.parents[:]
        random.shuffle(parents)
        return [(parents[i], parents[i+1]) for i in range(0, len(parents) - 1, 2)]

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
                    child1.variables[i] = avg
                    child2.variables[i] = avg
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
                children = []
                for _ in range(3):
                    child = p1.clone()
                    child.variables = []
                    for i in range(p1.num_of_variables):
                        x = p1.variables[i]
                        y = p2.variables[i]
                        # yrzy możliwe kombinacje
                        child1_val = 0.5 * x + 0.5 * y
                        child2_val = 1.5 * x - 0.5 * y
                        child3_val = -0.5 * x + 1.5 * y
                        child.variables.append(child1_val)
                    children.append(child)
                # wybieramy 2 pierwsze dzieci (wersja uproszczona)
                self.offspring.extend([children[0], children[1]])
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
                    c1.variables[i] = x + alpha * (y - x)
                    c2.variables[i] = y + alpha * (x - y)
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
                    c1.variables[i] = random.uniform(low, high)
                    c2.variables[i] = random.uniform(low, high)
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
