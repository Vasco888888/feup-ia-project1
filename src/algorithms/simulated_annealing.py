import numpy as np
import random
import math

def get_dist(i, j, problem, dm=None):
  if dm is not None:
    return dm[i, j]
  # On-the-fly calculation
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


def simulated_annealing(problem, initial_state=None, initial_temp=10000.0, cooling_rate=0.99, max_iterations_per_temp=None):
  """
  Simulated Annealing algorithm for the Santa 2012 TSP problem.
  Uses 2-opt moves, ensuring paths remain disjoint.
  """
  print("Simulated Annealing algorithm running...")
  
  n = len(problem.coords)
  # Precompute distance matrix only if memory allows (limit ~5000 cities)
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
  
  if max_iterations_per_temp is None:
    max_iterations_per_temp = min(n * 10, 5000)
  
  T = initial_temp
  T_min = 1e-3
  
  edges1, edges2 = edge_set(path1), edge_set(path2)
  current_obj = max(dist1, dist2)
  
  best_path1, best_path2 = path1.copy(), path2.copy()
  best_obj = current_obj
  
  iteration = 0
  while T > T_min:
    for _ in range(max_iterations_per_temp):
      which_path = random.choice([1, 2])
      
      # For 2-opt, we need at least 4 nodes
      if n < 4: break
      
      i = random.randint(0, n - 3)
      j = random.randint(i + 2, n - 1)
      if i == 0 and j == n - 1:
        continue
        
      if which_path == 1:
        a, b = path1[i], path1[i + 1]
        c, d = path1[j], path1[(j + 1) % n]
        
        # Check if new edges (a,c) and (b,d) conflict with path2
        if (min(a, c), max(a, c)) in edges2 or (min(b, d), max(b, d)) in edges2:
          continue
          
        delta = (get_dist(a, c, problem, dm) + get_dist(b, d, problem, dm)) - \
                (get_dist(a, b, problem, dm) + get_dist(c, d, problem, dm))
        new_dist1 = dist1 + delta
        new_dist2 = dist2
        new_obj = max(new_dist1, new_dist2)
        
        if new_obj < current_obj:
          accept = True
        else:
          try:
            accept = math.exp((current_obj - new_obj) / T) > random.random()
          except (OverflowError, ZeroDivisionError):
            accept = False
            
        if accept:
          path1[i + 1:j + 1] = path1[i + 1:j + 1][::-1]
          dist1 = new_dist1
          current_obj = new_obj
          
          edges1.remove((min(a, b), max(a, b)))
          edges1.remove((min(c, d), max(c, d)))
          edges1.add((min(a, c), max(a, c)))
          edges1.add((min(b, d), max(b, d)))
          
          if current_obj < best_obj:
            best_obj = current_obj
            best_path1, best_path2 = path1.copy(), path2.copy()
      else:
        a, b = path2[i], path2[i + 1]
        c, d = path2[j], path2[(j + 1) % n]
        
        if (min(a, c), max(a, c)) in edges1 or (min(b, d), max(b, d)) in edges1:
          continue
          
        delta = (get_dist(a, c, problem, dm) + get_dist(b, d, problem, dm)) - \
                (get_dist(a, b, problem, dm) + get_dist(c, d, problem, dm))
        new_dist1 = dist1
        new_dist2 = dist2 + delta
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
            
    T *= cooling_rate
    iteration += 1
    
    if iteration % 100 == 0:
      print(f"Temperature: {T:.4f} | Best Max Distance: {best_obj:.2f}")
    
  return best_path1, best_path2
