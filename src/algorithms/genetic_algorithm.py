import numpy as np

def _ox_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    n = len(parent1)
    a, b = sorted(np.random.choice(n, 2, replace=False))

    child = np.full(n, -1, dtype=int)
    child[a:b+1] = parent1[a:b+1]

    fill_vals = [v for v in parent2 if v not in child[a:b+1]]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill_vals[idx]
            idx += 1
    return child


def _get_edges(path: np.ndarray) -> set:
    n = len(path)
    return {(min(path[i], path[(i+1) % n]), max(path[i], path[(i+1) % n]))
            for i in range(n)}


def _repair_disjointness(path1: np.ndarray, path2: np.ndarray,
                          max_attempts: int = 200) -> tuple:
    p2 = path2.copy()
    edges1 = _get_edges(path1)

    for _ in range(max_attempts):
        shared = []
        n = len(p2)
        edges2 = _get_edges(p2)
        shared = edges1 & edges2
        if not shared:
            break

        u, v = next(iter(shared))
        pos = {city: i for i, city in enumerate(p2)}
        i, j = sorted([pos[u], pos[v]])
        if j - i > 1:
            p2[i:j+1] = p2[i:j+1][::-1]
        else:
            i, j = sorted(np.random.choice(n, 2, replace=False))
            p2[i:j+1] = p2[i:j+1][::-1]

    return path1, p2


def _two_opt_move(path: np.ndarray) -> np.ndarray:
    n = len(path)
    i, j = sorted(np.random.choice(n, 2, replace=False))
    new_path = path.copy()
    new_path[i:j+1] = path[i:j+1][::-1]
    return new_path


def _mutate(individual: tuple, mutation_rate: float) -> tuple:
    path1, path2 = individual

    if np.random.rand() < mutation_rate:
        path1 = _two_opt_move(path1)
    if np.random.rand() < mutation_rate:
        path2 = _two_opt_move(path2)

    path1, path2 = _repair_disjointness(path1, path2)
    return (path1, path2)


def _crossover(parent1: tuple, parent2: tuple) -> tuple:
    c1 = _ox_crossover(parent1[0], parent2[0])
    c2 = _ox_crossover(parent1[1], parent2[1])
    c1, c2 = _repair_disjointness(c1, c2)
    return (c1, c2)


def _tournament_select(population: list, fitnesses: np.ndarray,
                        k: int = 3) -> tuple:
    contestants = np.random.choice(len(population), k, replace=False)
    winner = contestants[np.argmin(fitnesses[contestants])]
    return population[winner]

def genetic_algorithm(
    problem,
    initial_state,
    population_size: int = 80,
    generations: int = 300,
    mutation_rate: float = 0.15,
    crossover_rate: float = 0.85,
    elitism: int = 5,
    tournament_k: int = 4,
    stagnation_limit: int = 60,
) -> tuple:
   
    print(f"GA | pop={population_size}  gens={generations}  "
          f"mut={mutation_rate}  cx={crossover_rate}")

    population = [initial_state]
    for _ in range(population_size - 1):
        ind = problem.get_random_solution()
        population.append(ind)

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

        if gen % 25 == 0 or stagnation == 0:
            print(f"  Gen {gen:3d} | best={best_fit:.2f}  stagnation={stagnation}")

        if stagnation >= stagnation_limit:
            print(f"  Early stop at gen {gen} (no improvement for {stagnation_limit} gens)")
            break

    print(f"\nGA finished — best distance: {best_fit:.2f}")
    return best_ind