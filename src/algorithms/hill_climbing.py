import numpy as np


def build_dist_matrix(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def path_length(path, dm):
    return dm[path, np.roll(path, -1)].sum()


def edge_set(path):
    n = len(path)
    return {(min(path[i], path[(i+1) % n]), max(path[i], path[(i+1) % n])) for i in range(n)}


def hill_climbing(problem, initial_state=None):
    dm = build_dist_matrix(problem.coords)
    path1, path2 = problem.get_random_solution() if initial_state is None else initial_state

    dist1, dist2 = path_length(path1, dm), path_length(path2, dm)
    n = len(path1)

    while True:
        edges1, edges2 = edge_set(path1), edge_set(path2)
        best_obj = max(dist1, dist2)   # current objective — only accept strict improvements
        best_move = None               # (i, j, which, delta)

        # 2-opt moves on path1 — new edges must not already exist in path2
        for i in range(n - 1):
            for j in range(i + 2, n):
                a, b = path1[i], path1[i + 1]
                c, d = path1[j], path1[(j + 1) % n]
                if (min(a, c), max(a, c)) in edges2 or (min(b, d), max(b, d)) in edges2:
                    continue
                delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])
                new_obj = max(dist1 + delta, dist2)
                if new_obj < best_obj:
                    best_obj, best_move = new_obj, (i, j, 'p1', delta)

        # 2-opt moves on path2 — new edges must not already exist in path1
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

        if best_move is None:           # local optimum reached
            return path1, path2

        i, j, which, delta = best_move
        if which == 'p1':
            path1 = path1.copy()
            path1[i + 1:j + 1] = path1[i + 1:j + 1][::-1]
            dist1 += delta
        else:
            path2 = path2.copy()
            path2[i + 1:j + 1] = path2[i + 1:j + 1][::-1]
            dist2 += delta

