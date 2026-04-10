import numpy as np
import math

def get_dist(i, j, problem, dm=None):
    """Retrieves distance from matrix or calculates it on the fly."""
    if dm is not None:
        return dm[i, j]
    c1 = problem.coords[i]
    c2 = problem.coords[j]
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def path_length(path, dm):
    """Calculates cycle length using precomputed distance matrix."""
    return dm[path, np.roll(path, -1)].sum()

def edge_set(path):
    """Converts a path into a set of undirected edges for constraint checking."""
    n = len(path)
    return {(min(path[i], path[(i+1) % n]), max(path[i], path[(i+1) % n])) for i in range(n)}

def hill_climbing(problem, initial_state=None):
    """
    Hill Climbing algorithm for the Santa 2012 TSP.
    
    Approach:
    - Moves: 2-opt swaps.
    - Style: Steepest Ascent (Exhaustive search of neighborhoods).
    - Termination: Stops at a Local Optimum (when no better 2-opt swap exists).
    """
    print("Hill Climbing algorithm running...")
    
    # Precompute distance matrix (only for small problems <= 100 cities based on main.py restriction)
    dm = None
    if len(problem.coords) <= 5000:
        diff = problem.coords[:, np.newaxis, :] - problem.coords[np.newaxis, :, :]
        dm = np.sqrt((diff ** 2).sum(axis=2))
    
    path1, path2 = problem.get_random_solution() if initial_state is None else initial_state

    dist1, dist2 = path_length(path1, dm), path_length(path2, dm)
    n = len(path1)

    while True:
        # Precompute current edge sets to validate moves efficiently
        edges1, edges2 = edge_set(path1), edge_set(path2)
        best_obj = max(dist1, dist2)
        best_move = None               # (i, j, which_path, delta)

        # 2-OPT NEIGHBORHOOD SEARCH (Path 1)
        # O(N^2) complexity: checks all possible edge swaps
        for i in range(n - 1):
            for j in range(i + 2, n):
                a, b = path1[i], path1[i + 1]
                c, d = path1[j], path1[(j + 1) % n]
                
                # Check if proposed edges in Path 1 already exist in Path 2
                if (min(a, c), max(a, c)) in edges2 or (min(b, d), max(b, d)) in edges2:
                    continue
                    
                # Calculate improvement locally
                delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])
                new_obj = max(dist1 + delta, dist2)
                if new_obj < best_obj:
                    best_obj, best_move = new_obj, (i, j, 'p1', delta)

        # 2-OPT NEIGHBORHOOD SEARCH (Path 2)
        for i in range(n - 1):
            for j in range(i + 2, n):
                a, b = path2[i], path2[i + 1]
                c, d = path2[j], path2[(j + 1) % n]
                
                if (min(a, c), max(a, c)) in edges1 or (min(b, d), max(b, d)) in edges1:
                    continue
                    
                delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])
                new_obj = max(dist1, dist2 + delta)
                if new_obj < best_obj:
                    best_obj, best_move = new_obj, (i, j, 'p2', delta)

        # TERMINATION: If no improving move was found in either neighborhood, we are at a Local Optimum
        if best_move is None:
            return path1, path2

        # APPLY BEST MOVE
        i, j, which, delta = best_move
        if which == 'p1':
            path1 = path1.copy()
            path1[i + 1:j + 1] = path1[i + 1:j + 1][::-1]
            dist1 += delta
        else:
            path2 = path2.copy()
            path2[i + 1:j + 1] = path2[i + 1:j + 1][::-1]
            dist2 += delta

        # Progress reporting
        print(f"  [HC] Best Max Distance: {best_obj:.2f}")
