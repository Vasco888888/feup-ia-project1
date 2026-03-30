import numpy as np
import random
import math

def build_dist_matrix(coords):
  diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
  return np.sqrt((diff ** 2).sum(axis=2))

def path_length(path, dm):
  return dm[path, np.roll(path, -1)].sum()

def edge_set(path):
  n = len(path)
  return {(min(path[i], path[(i+1) % n]), max(path[i], path[(i+1) % n])) for i in range(n)}


def simulated_annealing(problem, initial_state=None, initial_temp=10000.0, cooling_rate=0.99, max_iterations_per_temp=None):
  """
  Simulated Annealing algorithm for the Santa 2012 TSP problem.
  Uses 2-opt moves, ensuring paths remain disjoint.
  """
  print("Simulated Annealing algorithm running...")
  dm = build_dist_matrix(problem.coords)
  path1, path2 = problem.get_random_solution() if initial_state is None else initial_state

  dist1, dist2 = path_length(path1, dm), path_length(path2, dm)
  n = len(path1)
  
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
      
      i = random.randint(0, n - 3)
      j = random.randint(i + 2, n - 1)
      if i == 0 and j == n - 1:
        continue
        
      if which_path == 1:
        a, b = path1[i], path1[i + 1]
        c, d = path1[j], path1[(j + 1) % n]
        
        if (min(a, c), max(a, c)) in edges2 or (min(b, d), max(b, d)) in edges2:
          continue
          
        delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])
        new_dist1 = dist1 + delta
        new_dist2 = dist2
        new_obj = max(new_dist1, new_dist2)
        
        # If delta is large enough to cause exp overflow or extremely small, we safeguard it
        if new_obj < current_obj:
          accept = True
        else:
          try:
            accept = math.exp((current_obj - new_obj) / T) > random.random()
          except OverflowError:
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
          
        delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])
        new_dist1 = dist1
        new_dist2 = dist2 + delta
        new_obj = max(new_dist1, new_dist2)
        
        if new_obj < current_obj:
          accept = True
        else:
          try:
            accept = math.exp((current_obj - new_obj) / T) > random.random()
          except OverflowError:
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
    
    # Print progress every few temperature drops
    if iteration % 100 == 0:
      print(f"Temperature: {T:.4f} | Best Max Distance: {best_obj:.2f}")
    
  return best_path1, best_path2
