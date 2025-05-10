import numpy as np
import benchmark_functions as bf
from opfunu import cec_based

# Wybierz funkcję celu tutaj:
FUNCTION_NAME = "hyperellipsoid"   # lub: "cec_f3"

NUM_DIMENSIONS = 10

if FUNCTION_NAME == "hyperellipsoid":
    func = bf.Hyperellipsoid(n_dimensions=NUM_DIMENSIONS)
    bounds = func.suggested_bounds()
    def evaluate_fitness(x):
        return -func._evaluate(x)
elif FUNCTION_NAME == "cec_f3":
    func = cec_based.cec2014.F32014(ndim=NUM_DIMENSIONS)
    bounds = func.bounds
    def evaluate_fitness(x):
        return -func.evaluate(x)
else:
    raise ValueError("Nieznana funkcja celu!")

def get_bounds():
    return bounds

def get_dimensions():
    return NUM_DIMENSIONS
