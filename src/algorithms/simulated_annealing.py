import numpy as np
import random
import math

def get_dist(i, j, problem, dm=None):
  """Retrieves distance from matrix or calculates it on the fly (memory safety)."""
  if dm is not None:
    return dm[i, j]
  c1 = problem.coords[i]
  c2 = problem.coords[j]
  return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def calculate_full_path_length(path, problem, dm=None):
  """Calculates total cycle length using the memory-aware distance getter."""
  length = 0
  n = len(path)
  for i in range(n):
    length += get_dist(path[i], path[(i + 1) % n], problem, dm)
  return length

def edge_set(path):
  """Returns a set of sorted edge tuples to represent the path's connectivity."""
  n = len(path)
  return {(min(path[i], path[(i+1) % n]), max(path[i], path[(i+1) % n])) for i in range(n)}

def simulated_annealing(problem, initial_state=None, initial_temp=10000.0, cooling_rate=0.99, max_iterations_per_temp=None):
  """
  Simulated Annealing metaheuristic for the Santa 2012 TSP.
  
  Approach:
  - Moves: Probabilistic acceptance of 2-opt swaps.
  - Constraints: Moves that break path disjointness (edge collisions) are rejected immediately.
  - Evaluation: Minimizes the maximum length of the two paths (min-max).
  """
  print("Simulated Annealing algorithm running...")
  
  n = len(problem.coords)
  
  # Adaptive memory management: precompute matrix only if small enough
  dm = None
  if n <= 5000:
    print(f"Precomputing distance matrix for {n} cities...")
    diff = problem.coords[:, np.newaxis, :] - problem.coords[np.newaxis, :, :]
    dm = np.sqrt((diff ** 2).sum(axis=2))
  else:
    print(f"Dataset too large ({n} cities), calculating distances on-the-fly.")

  # Starting state: use provided initial state or generate random
  path1, path2 = problem.get_random_solution() if initial_state is None else initial_state

  # Initial distances and tracking sets
  dist1 = calculate_full_path_length(path1, problem, dm)
  dist2 = calculate_full_path_length(path2, problem, dm)
  
  # Number of attempts per temperature step scales with problem size
  if max_iterations_per_temp is None:
    max_iterations_per_temp = min(n * 10, 5000)
  
  T = initial_temp
  T_min = 1e-3 # Stop at nearly zero degrees
  
  # Active edge tracking for disjointness validation
  edges1, edges2 = edge_set(path1), edge_set(path2)
  current_obj = max(dist1, dist2)
  
  best_path1, best_path2 = path1.copy(), path2.copy()
  best_obj = current_obj
  
  iteration = 0
  while T > T_min:
    for _ in range(max_iterations_per_temp):
      # Pick one of the two paths to attempt a local move
      which_path = random.choice([1, 2])
      
      if n < 4: break # 2-opt requires at least 4 cities
      
      i = random.randint(0, n - 3)
      j = random.randint(i + 2, n - 1)
      if i == 0 and j == n - 1: continue # Avoid trivial whole-path reverses
        
      if which_path == 1:
        # 2-opt move on Path 1: remove edges (a,b), (c,d) -> add (a,c), (b,d)
        a, b = path1[i], path1[i + 1]
        c, d = path1[j], path1[(j + 1) % n]
        
        # DISJOINT CONSTRAINT: Check if new edges for Path 1 exist in Path 2
        if (min(a, c), max(a, c)) in edges2 or (min(b, d), max(b, d)) in edges2:
          continue
         
        # Calculate change in length locally (O(1) instead of O(N))
        delta = (get_dist(a, c, problem, dm) + get_dist(b, d, problem, dm)) - \
            (get_dist(a, b, problem, dm) + get_dist(c, d, problem, dm))
        new_dist1 = dist1 + delta
        new_dist2 = dist2
        new_obj = max(new_dist1, new_dist2)
        
        # METROPOLIS CRITERION: Accept if better, or with probability based on T
        if new_obj < current_obj:
          accept = True
        else:
          try:
            accept = math.exp((current_obj - new_obj) / T) > random.random()
          except (OverflowError, ZeroDivisionError):
            accept = False
          
        if accept:
          # In-place 2-opt reversal
          path1[i + 1:j + 1] = path1[i + 1:j + 1][::-1]
          dist1 = new_dist1
          current_obj = new_obj
          
          # Update edge set incrementally
          edges1.remove((min(a, b), max(a, b)))
          edges1.remove((min(c, d), max(c, d)))
          edges1.add((min(a, c), max(a, c)))
          edges1.add((min(b, d), max(b, d)))
          
          # Check for global best
          if current_obj < best_obj:
            best_obj = current_obj
            best_path1, best_path2 = path1.copy(), path2.copy()
      else:
        # Symmetric logic for Path 2...
        a, b = path2[i], path2[i + 1]
        c, d = path2[j], path2[(j + 1) % n]
        
        if (min(a, c), max(a, c)) in edges1 or (min(b, d), max(b, d)) in edges1:
          continue
         
        delta = (get_dist(a, c, problem, dm) + get_dist(b, d, problem, dm)) - \
            (get_dist(a, b, problem, dm) + get_dist(c, d, problem, dm))
        new_dist2 = dist2 + delta
        new_dist1 = dist1
        new_obj = max(new_dist1, new_dist2)
        
        if new_obj < current_obj:
          accept = True
        else:
          try:
            accept = math.exp((current_obj - new_obj) / T) > random.random()
          except (OverflowError, ZeroDivisionError):
            accept = False
          
        if accept:
          path2[i + 1:j + 1] = path2[i + 1:j + 1][::-1]
          dist2 = new_dist2
          current_obj = new_obj
          
          edges2.remove((min(a, b), max(a, b)))
          edges2.remove((min(c, d), max(c, d)))
          edges2.add((min(a, c), max(a, c)))
          edges2.add((min(b, d), max(b, d)))
          
          if current_obj < best_obj:
            best_obj = current_obj
            best_path1, best_path2 = path1.copy(), path2.copy()
          
    # Cooling schedule: geometric reduction
    T *= cooling_rate
    iteration += 1
    
    if iteration % 100 == 0:
      print(f"  [SA] Temperature: {T:.4f} | Best Max Distance: {best_obj:.2f}")
    
  return best_path1, best_path2
