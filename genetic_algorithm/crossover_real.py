import random
from .chromosome_real import ChromosomeReal


class CrossoverReal:
    def __init__(self, parents, crossover_probability):
        """
        parents: lista osobników typu ChromosomeReal
        crossover_probability: prawdopodobieństwo krzyżowania
        """
        self.parents = parents
        self.crossover_probability = crossover_probability
        self.offspring = []

    def _prepare_pairs(self):
        """Losowe parowanie osobników do krzyżowania"""
        parents = self.parents[:]
        random.shuffle(parents)
        return [(parents[i], parents[i+1]) for i in range(0, len(parents) - 1, 2)]

    def arithmetic_crossover(self):
        """Krzyżowanie arytmetyczne: średnia z wartości rodziców"""
        self.offspring = []
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
        return self.offspring

    def linear_crossover(self):
        """
        Krzyżowanie liniowe:
        Tworzy 3 dzieci: 
        c1 = 0.5*p1 + 0.5*p2
        c2 = 1.5*p1 - 0.5*p2
        c3 = -0.5*p1 + 1.5*p2
        Wybieramy 2 najlepsze losowo
        """
        self.offspring = []
        for p1, p2 in self._prepare_pairs():
            if random.random() <= self.crossover_probability:
                children = []
                for _ in range(3):
                    child = p1.clone()
                    child.variables = []
                    for i in range(p1.num_of_variables):
                        x = p1.variables[i]
                        y = p2.variables[i]
                        # Trzy możliwe kombinacje
                        child1_val = 0.5 * x + 0.5 * y
                        child2_val = 1.5 * x - 0.5 * y
                        child3_val = -0.5 * x + 1.5 * y
                    children.append(child1_val)
                # Uproszczona wersja: wybieramy dwa pierwsze dzieci
                c1 = p1.clone()
                c1.variables = [(0.5 * x + 0.5 * y) for x, y in zip(p1.variables, p2.variables)]
                c2 = p1.clone()
                c2.variables = [(1.5 * x - 0.5 * y) for x, y in zip(p1.variables, p2.variables)]
                self.offspring.extend([c1, c2])
            else:
                self.offspring.extend([p1.clone(), p2.clone()])
        return self.offspring

    def alpha_blend_crossover(self, alpha=0.5):
        """
        Krzyżowanie mieszające typu alfa:
        Dla każdego genu:
        dziecko = p1 + alpha * (p2 - p1)
        """
        self.offspring = []
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
        return self.offspring

    def alpha_beta_blend_crossover(self, alpha=0.5, beta=0.5):
        """
        Krzyżowanie mieszające typu alfa-beta:
        dziecko = losowa wartość z przedziału:
        [min(x, y) - alpha * d, max(x, y) + beta * d]
        gdzie d = |x - y|
        """
        self.offspring = []
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
        return self.offspring

    def average_crossover(self):
        """
        Krzyżowanie uśredniające:
        Dla każdego genu: dziecko = (p1 + p2) / 2
        (to samo co arytmetyczne, powtórka tylko dla formalności)
        """
        return self.arithmetic_crossover()
