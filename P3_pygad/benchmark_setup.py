import benchmark_functions as bf
from opfunu import cec_based

# Domyślne wartości
FUNCTION_NAME = "hyperellipsoid"
NUM_DIMENSIONS = 10
bounds = [(-100, 100)]
evaluate_fitness = lambda x: 0  # placeholder

def set_function(name):
    global FUNCTION_NAME, NUM_DIMENSIONS, bounds, evaluate_fitness

    FUNCTION_NAME = name
    NUM_DIMENSIONS = 10  # Można też dynamicznie ustawiać, np. w GUI

    if FUNCTION_NAME == "hyperellipsoid":
        func = bf.Hyperellipsoid(n_dimensions=NUM_DIMENSIONS)
        bounds = func.suggested_bounds()
        evaluate_fitness = lambda x: -func._evaluate(x)

    elif FUNCTION_NAME == "cec_f3":
        func = cec_based.cec2014.F32014(ndim=NUM_DIMENSIONS)
        bounds = func.bounds
        evaluate_fitness = lambda x: -func.evaluate(x)

    else:
        raise ValueError("Nieznana funkcja celu: " + name)

def get_bounds():
    return bounds

def get_dimensions():
    return NUM_DIMENSIONS
