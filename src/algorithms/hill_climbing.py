import numpy as np
import math

def get_dist(i, j, problem, dm=None):
  if dm is not None:
    return dm[i, j]
  c1 = problem.coords[i]
  c2 = problem.coords[j]
  return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def calculate_full_path_length(path, problem, dm=None):
  length = 0
  n = len(path)
  for i in range(n):
    length += get_dist(path[i], path[(i + 1) % n], problem, dm)
  return length

def edge_set(path):
  n = len(path)
  return {(min(path[i], path[(i+1) % n]), max(path[i], path[(i+1) % n])) for i in range(n)}


def hill_climbing(problem, initial_state=None):
  """
  Hill Climbing algorithm for the Santa 2012 TSP problem.
  Uses 2-opt moves, ensuring paths remain disjoint.
  """
  print("Hill Climbing algorithm running...")
  
  n = len(problem.coords)
  dm = None
  if n <= 5000:
    print(f"Precomputing distance matrix for {n} cities...")
    diff = problem.coords[:, np.newaxis, :] - problem.coords[np.newaxis, :, :]
    dm = np.sqrt((diff ** 2).sum(axis=2))
  else:
    print(f"Dataset too large ({n} cities), calculating distances on-the-fly.")

  path1, path2 = problem.get_random_solution() if initial_state is None else initial_state

  dist1 = calculate_full_path_length(path1, problem, dm)
  dist2 = calculate_full_path_length(path2, problem, dm)

  while True:
    edges1, edges2 = edge_set(path1), edge_set(path2)
    best_obj = max(dist1, dist2)
    best_move = None               # (i, j, which, delta)

    # 2-opt moves on path1
    for i in range(n - 1):
      for j in range(i + 2, n):
        a, b = path1[i], path1[i + 1]
        c, d = path1[j], path1[(j + 1) % n]
        if (min(a, c), max(a, c)) in edges2 or (min(b, d), max(b, d)) in edges2:
          continue
        # Use on-the-fly calculation
        a_b = get_dist(a, b, problem, dm)
        c_d = get_dist(c, d, problem, dm)
        a_c = get_dist(a, c, problem, dm)
        b_d = get_dist(b, d, problem, dm)
        delta = (a_c + b_d) - (a_b + c_d)
        new_obj = max(dist1 + delta, dist2)
        if new_obj < best_obj:
          best_obj, best_move = new_obj, (i, j, 'p1', delta)

    # 2-opt moves on path2
    for i in range(n - 1):
      for j in range(i + 2, n):
        a, b = path2[i], path2[i + 1]
        c, d = path2[j], path2[(j + 1) % n]
        if (min(a, c), max(a, c)) in edges1 or (min(b, d), max(b, d)) in edges1:
          continue
        a_b = get_dist(a, b, problem, dm)
        c_d = get_dist(c, d, problem, dm)
        a_c = get_dist(a, c, problem, dm)
        b_d = get_dist(b, d, problem, dm)
        delta = (a_c + b_d) - (a_b + c_d)
        new_obj = max(dist1, dist2 + delta)
        if new_obj < best_obj:
          best_obj, best_move = new_obj, (i, j, 'p2', delta)

    if best_move is None:
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
