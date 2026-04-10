import time
import numpy as np

# Dataset size threshold for switching between robust randomized repair and fast mathematical repair
FAST_DISJOINT_THRESHOLD = 1_000

def _build_dist_matrix(coords: np.ndarray) -> np.ndarray:
    """Precomputes an NxN Euclidean distance matrix for vectorized calculations."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _path_length_dm(path: np.ndarray, dm, coords=None) -> float:
    """Calculates path distance using a distance matrix or coordinates (for memory efficiency)."""
    if dm is not None:
        return float(dm[path, np.roll(path, -1)].sum())
    ordered = coords[path]
    next_ordered = coords[np.roll(path, -1)]
    return float(np.sqrt(((ordered - next_ordered) ** 2).sum(axis=1)).sum())


def _fitness(ind: tuple, dm, coords=None) -> float:
    """The fitness of an individual is the maximum of its two path lengths (minimax objective)."""
    return max(_path_length_dm(ind[0], dm, coords), _path_length_dm(ind[1], dm, coords))


def _find_coprime_step(n: int) -> int:
    """
    Finds a step 'k' coprime to 'n' (gcd(k,n)=1).
    Traversing n items with step k forms a Hamiltonian cycle.
    """
    if n < 5:
        return 1

    for step in (3, 5, 7, 9, 11, 13):
        if step < n and np.gcd(step, n) == 1:
            return step

    for step in range(2, n - 1):
        if np.gcd(step, n) == 1 and step != 1 and step != n - 1:
            return step
    return 1


def _build_disjoint_cycle_by_step(path1: np.ndarray) -> np.ndarray:
    """
    Rapidly constructs a disjoint cycle using a mathematical coprime step.
    Ensures Path 2 does not share edges with Path 1.
    """
    n = len(path1)
    if n < 5:
        return path1.copy()

    k = _find_coprime_step(n)
    idx = (np.arange(n, dtype=int) * k) % n
    return path1[idx]


def _build_disjoint_path(path1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generates a second path that shares no edges with the first.
    Uses randomized repair for small N and mathematical repair for large N.
    """
    n = len(path1)
    if n >= FAST_DISJOINT_THRESHOLD:
        return _build_disjoint_cycle_by_step(path1)

    lo = n // 4
    hi = 3 * n // 4
    path2 = path1.copy()
    path2[lo:hi] = path1[lo:hi][::-1]

    edges1 = _edge_set(path1)
    for _ in range(n * 2):
        shared = _shared_edges(path2, edges1)
        if not shared:
            break
        i = shared[rng.integers(len(shared))]
        j = (i + 1) % n
        if j > i:
            lo2 = i + 1
            hi2 = rng.integers(lo2 + 1, n + 1)
            path2[lo2:hi2] = path2[lo2:hi2][::-1]
        else:
            path2 = np.roll(path2, -(i + 1))

    return path2


def _edge_set(path: np.ndarray) -> set:
    """Returns a set of undirected edges (sorted tuples) for the path."""
    n = len(path)
    return {(min(path[i], path[(i + 1) % n]), max(path[i], path[(i + 1) % n]))
            for i in range(n)}


def _shared_edges(path2: np.ndarray, edges1: set) -> list:
    """Identifies indices in path2 where the edges conflict with Path 1."""
    n = len(path2)
    result = []
    for i in range(n):
        j = (i + 1) % n
        e = (min(path2[i], path2[j]), max(path2[i], path2[j]))
        if e in edges1:
            result.append(i)
    return result


def _is_disjoint(path1: np.ndarray, path2: np.ndarray) -> bool:
    """Checks if the two cycles share no common edges."""
    return len(_shared_edges(path2, _edge_set(path1))) == 0


def _repair_disjoint_path(path1: np.ndarray, path2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Repairs broken disjoint path constraints after genetic operations.
    Attempts local reversals first, then falls back to coprime step if needed.
    """
    if _is_disjoint(path1, path2):
        return path2

    n = len(path1)
    if n >= FAST_DISJOINT_THRESHOLD:
        return _build_disjoint_cycle_by_step(path1)

    edges1 = _edge_set(path1)
    candidate = path2.copy()

    for _ in range(min(200, max(50, n // 2))):
        shared = _shared_edges(candidate, edges1)
        if not shared:
            return candidate
        candidate = _two_opt_move(candidate, rng)

    fallback = _build_disjoint_path(path1, rng)
    if _is_disjoint(path1, fallback):
        return fallback

    for _ in range(max(100, n)):
        perm = rng.permutation(n)
        if _is_disjoint(path1, perm):
            return perm

    return fallback


def _ox_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ordered Crossover (OX) for Permutation problems: preserves relative order of cities."""
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    child = np.full(n, -1, dtype=int)
    child[a:b + 1] = p1[a:b + 1]
    in_slice = set(p1[a:b + 1])
    fill = [v for v in p2 if v not in in_slice]
    ptr = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[ptr]
            ptr += 1
    return child


def _crossover(par1: tuple, par2: tuple, rng: np.random.Generator) -> tuple:
    """Executes crossover on both paths while enforcing the disjoint constraint."""
    child_p1 = _ox_crossover(par1[0], par2[0], rng)
    child_p2_candidate = _ox_crossover(par1[1], par2[1], rng)
    child_p2 = _repair_disjoint_path(child_p1, child_p2_candidate, rng)
    return (child_p1, child_p2)


def _two_opt_move(path: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Performs a random 2-opt reversal (edge swap)."""
    n = len(path)
    i, j = sorted(rng.choice(n, 2, replace=False))
    new_path = path.copy()
    new_path[i:j + 1] = path[i:j + 1][::-1]
    return new_path


def _mutate(ind: tuple, mutation_rate: float, rng: np.random.Generator) -> tuple:
    """Apply mutations to individuals to maintain genetic diversity."""
    path1, path2 = ind
    mutated_p1 = rng.random() < mutation_rate
    mutated_p2 = rng.random() < mutation_rate

    if mutated_p1:
        path1 = _two_opt_move(path1, rng)
        path2 = _build_disjoint_path(path1, rng)
    elif mutated_p2:
        candidate = _two_opt_move(path2, rng)
        path2 = _repair_disjoint_path(path1, candidate, rng)

    if not _is_disjoint(path1, path2):
        path2 = _repair_disjoint_path(path1, path2, rng)

    return (path1, path2)


def _tournament_select(
    population: list,
    fitnesses: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple:
    """Tournament Selection: standard method for choosing parents from a population."""
    k = min(k, len(population))
    contestants = rng.choice(len(population), k, replace=False)
    winner = contestants[int(np.argmin(fitnesses[contestants]))]
    return population[winner]

def _scale_params(num_cities: int):
    """Adaptive parameter scaling for population size and generations based on dataset size."""
    if num_cities <= 100:
        return 60, 300, 60
    elif num_cities <= 1_000:
        return 40, 150, 40
    elif num_cities <= 10_000:
        return 20, 80, 25
    else:
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
    seed: int = None,
    max_runtime_s: float = None,
) -> tuple:
    """
    Genetic Algorithm for the Santa 2012 dual-path TSP problem.
    """
    rng = np.random.default_rng(seed)

    # Automatically scale population/generations if not provided
    pop_auto, gen_auto, stag_auto = _scale_params(problem.num_cities)
    if population_size is None:
        population_size = pop_auto
    if generations is None:
        generations = gen_auto
    if stagnation_limit is None:
        stagnation_limit = stag_auto

    # Prevent project evaluation timeouts on large datasets
    if max_runtime_s is None:
        if problem.num_cities >= 10_000:
            max_runtime_s = 120.0
        elif problem.num_cities >= 1_000:
            max_runtime_s = 90.0

    print("Genetic Algorithm running...")

    coords = problem.coords
    # Only precompute distance matrix for small/medium problems to avoid MemoryError
    if problem.num_cities <= 5_000:
        dm = _build_dist_matrix(coords)
    else:
        dm = None

    # Population Initialization
    population: list[tuple] = [initial_state]
    for _ in range(population_size - 1):
        p1 = rng.permutation(problem.num_cities)
        p2 = _build_disjoint_path(p1, rng)
        if not _is_disjoint(p1, p2):
            p2 = _repair_disjoint_path(p1, p2, rng)
        population.append((p1, p2))

    fitnesses = np.array([_fitness(ind, dm, coords) for ind in population])

    # Best-so-far tracking
    best_idx = int(np.argmin(fitnesses))
    best_ind = population[best_idx]
    best_fit = fitnesses[best_idx]
    stagnation = 0
    ga_start_time = time.time()
    progress_every = max(1, generations // 5)

    for gen in range(1, generations + 1):
        
        # Elitism: Keep the best performers
        order = np.argsort(fitnesses)
        elite_inds = [(population[i][0].copy(), population[i][1].copy()) for i in order[:elitism]]
        elite_fits = fitnesses[order[:elitism]]

        new_population: list[tuple] = elite_inds
        new_fitnesses: list[float] = list(elite_fits)

        # Reproduction Loop
        while len(new_population) < population_size:
            p1 = _tournament_select(population, fitnesses, tournament_k, rng)
            if rng.random() < crossover_rate:
                p2 = _tournament_select(population, fitnesses, tournament_k, rng)
                child = _crossover(p1, p2, rng)
            else:
                child = (p1[0].copy(), p1[1].copy())

            child = _mutate(child, mutation_rate, rng)
            new_population.append(child)
            new_fitnesses.append(_fitness(child, dm, coords))

        population = new_population
        fitnesses = np.array(new_fitnesses)

        # Progress tracking and convergence monitor
        gen_best_idx = int(np.argmin(fitnesses))
        gen_best_fit = fitnesses[gen_best_idx]

        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best_ind = population[gen_best_idx]
            stagnation = 0
        else:
            stagnation += 1

        if stagnation >= stagnation_limit:
            break

        if problem.num_cities >= 1_000 and (gen == 1 or gen % progress_every == 0):
            print(f"  [GA] gen={gen}/{generations} best={best_fit:.2f} stagnation={stagnation}")

        # Safety break for time limits
        if max_runtime_s is not None:
            now_s = time.time()
            if now_s - ga_start_time >= max_runtime_s:
                print(f"  [GA] runtime limit reached ({max_runtime_s:.0f}s), returning best-so-far.")
                break

    # Ensure final output is fully valid
    best_p1, best_p2 = best_ind
    if not _is_disjoint(best_p1, best_p2):
        best_p2 = _repair_disjoint_path(best_p1, best_p2, rng)

    return (best_p1, best_p2)