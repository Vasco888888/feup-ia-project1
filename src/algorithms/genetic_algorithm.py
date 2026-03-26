import numpy as np

def _ox_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    n = len(parent1)
    a, b = sorted(np.random.choice(n, 2, replace=False))

    child = np.full(n, -1, dtype=int)
    child[a:b+1] = parent1[a:b+1]

    in_slice = set(parent1[a:b+1])
    fill_vals = [v for v in parent2 if v not in in_slice]

    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill_vals[idx]
            idx += 1
    return child


def _build_disjoint_path(path1: np.ndarray) -> np.ndarray:
    n = len(path1)
    lo = max(1, n // 4)
    hi = max(lo + 1, 3 * n // 4)
    offset = np.random.randint(lo, hi)
    return np.roll(path1, offset)


def _two_opt_move(path: np.ndarray) -> np.ndarray:
    n = len(path)
    i, j = sorted(np.random.choice(n, 2, replace=False))
    new_path = path.copy()
    new_path[i:j+1] = path[i:j+1][::-1]
    return new_path


def _mutate(individual: tuple, mutation_rate: float) -> tuple:
    path1, path2 = individual
    mutated_p1 = False

    if np.random.rand() < mutation_rate:
        path1 = _two_opt_move(path1)
        mutated_p1 = True

    if mutated_p1:
        path2 = _build_disjoint_path(path1)
    elif np.random.rand() < mutation_rate:
        path2 = _two_opt_move(path2)

    return (path1, path2)


def _crossover(parent1: tuple, parent2: tuple) -> tuple:
    child_p1 = _ox_crossover(parent1[0], parent2[0])
    child_p2 = _build_disjoint_path(child_p1)
    return (child_p1, child_p2)


def _tournament_select(population: list, fitnesses: np.ndarray, k: int) -> tuple:
    contestants = np.random.choice(len(population), k, replace=False)
    winner = contestants[np.argmin(fitnesses[contestants])]
    return population[winner]


def _scale_params(num_cities: int):
    if num_cities <= 100:
        return 60, 300, 60
    elif num_cities <= 1000:
        return 40, 150, 40
    elif num_cities <= 10000:
        return 20, 80, 25
    else:  # 150000
        return 10, 30, 15


def genetic_algorithm(
    problem,
    initial_state,
    population_size: int = None,
    generations: int = None,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.85,
    elitism: int = 2,
    tournament_k: int = 3,
    stagnation_limit: int = None,
) -> tuple:
    
    pop_auto, gen_auto, stag_auto = _scale_params(problem.num_cities)
    if population_size is None:
        population_size = pop_auto
    if generations is None:
        generations = gen_auto
    if stagnation_limit is None:
        stagnation_limit = stag_auto

    print(f"GA | cities={problem.num_cities}  pop={population_size}  "
          f"gens={generations}  mut={mutation_rate}  cx={crossover_rate}")

    population = [initial_state]
    for _ in range(population_size - 1):
        p1 = np.random.permutation(problem.num_cities)
        p2 = _build_disjoint_path(p1)
        population.append((p1, p2))

    fitnesses = np.array([problem.calculate_distance(ind) for ind in population])

    best_idx = int(np.argmin(fitnesses))
    best_ind = population[best_idx]
    best_fit = fitnesses[best_idx]
    stagnation = 0

    print(f"  Gen   0 | best={best_fit:.2f}")

    for gen in range(1, generations + 1):

        order = np.argsort(fitnesses)
        new_population = [population[i] for i in order[:elitism]]

        while len(new_population) < population_size:
            p1 = _tournament_select(population, fitnesses, tournament_k)
            if np.random.rand() < crossover_rate:
                p2 = _tournament_select(population, fitnesses, tournament_k)
                child = _crossover(p1, p2)
            else:
                child = (p1[0].copy(), p1[1].copy())

            child = _mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        fitnesses = np.array([problem.calculate_distance(ind) for ind in population])

        gen_best_idx = int(np.argmin(fitnesses))
        gen_best_fit = fitnesses[gen_best_idx]

        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best_ind = population[gen_best_idx]
            stagnation = 0
        else:
            stagnation += 1

        if gen % 10 == 0 or stagnation == 0:
            print(f"  Gen {gen:3d} | best={best_fit:.2f}  stagnation={stagnation}")

        if stagnation >= stagnation_limit:
            print(f"  Early stop at gen {gen} (no improvement for {stagnation_limit} gens)")
            break

    print(f"\nGA finished — best distance: {best_fit:.2f}")
    return best_ind