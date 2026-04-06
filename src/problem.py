import numpy as np
import pandas as pd
from src.utils import load_cities

class SantaProblem:
  def __init__(self, size):
    self.size = size
    self.cities_df = load_cities(size)
    
    # Detect coordinate columns
    x_col = 'X' if 'X' in self.cities_df.columns else 'x'
    y_col = 'Y' if 'Y' in self.cities_df.columns else 'y'
    self.id_col = 'CityId' if 'CityId' in self.cities_df.columns else 'id'
    
    self.coords = self.cities_df[[x_col, y_col]].values
    self.num_cities = len(self.cities_df)

  def _path_distance(self, path):
    """Calculates the standard Euclidean distance of a single closed path."""
    ordered_coords = self.coords[path]
    diffs = np.diff(ordered_coords, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    
    # Add distance from last city back to Depot (City 0)
    last_to_depot = np.sqrt(np.sum((ordered_coords[-1] - self.coords[path[0]])**2))
    return np.sum(distances) + last_to_depot

  def calculate_distance(self, paths):
    """
    In the 2012 version, the objective is to minimize the max distance 
    of the two disjoint paths.
    """
    path1, path2 = paths
    dist1 = self._path_distance(path1)
    dist2 = self._path_distance(path2)
    return max(dist1, dist2)

  def validate_path(self, paths):
    """
    Checks if two paths are valid for Santa's 2012 TSP.
    They must visit all cities and be disjoint (share no edges).
    """
    if len(paths) != 2:
      return False, "Expected a tuple of two paths"
      
    path1, path2 = paths
    
    for p_idx, path in enumerate([path1, path2]):
      if len(path) != self.num_cities:
        return False, f"Wrong path length for Path {p_idx+1}: {len(path)} (expected {self.num_cities})"
      unique_cities = set(path)
      if len(unique_cities) != self.num_cities:
        return False, f"Duplicate or missing cities in Path {p_idx+1}"
        
    # Check disjointness
    edges1 = set()
    for i in range(len(path1)):
      u, v = path1[i], path1[(i+1)%len(path1)]
      edges1.add((min(u, v), max(u, v)))
      
    for i in range(len(path2)):
      u, v = path2[i], path2[(i+1)%len(path2)]
      edge = (min(u, v), max(u, v))
      if edge in edges1:
        return False, f"Paths are not disjoint, they share edge {edge}"
        
    return True, "Valid Santa 2012 Paths"

  def plot_solution(self, paths, title="Santa TSP 2012 Solution"):
    """
    Plots the cities and the two paths connecting them.
    """
    import matplotlib.pyplot as plt
    
    path1, path2 = paths
    
    ordered_coords1 = self.coords[path1]
    loop_coords1 = np.vstack([ordered_coords1, self.coords[path1[0]]])
    
    ordered_coords2 = self.coords[path2]
    loop_coords2 = np.vstack([ordered_coords2, self.coords[path2[0]]])
    
    plt.figure(figsize=(10, 8))
    plt.plot(loop_coords1[:, 0], loop_coords1[:, 1], 'b-', alpha=0.6, label='Path 1')
    plt.plot(loop_coords2[:, 0], loop_coords2[:, 1], 'g-', alpha=0.6, label='Path 2')
    plt.scatter(self.coords[:, 0], self.coords[:, 1], c='red', s=10, label='Cities')
    
    dist1 = self._path_distance(path1)
    dist2 = self._path_distance(path2)
    
    plt.title(f"{title}\n(Max Distance: {max(dist1, dist2):.2f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.savefig('solution.png')
    print(f"Plot saved to solution.png")
    plt.close()

  def get_random_solution(self):
    """
    Generates two random valid disjoint paths.
    """
    path1 = np.random.permutation(self.num_cities)
    
    if self.num_cities < 5:
      # Cannot have 2 disjoint Hamiltonian cycles for N < 5
      return (path1, np.random.permutation(self.num_cities))
      
    edges1 = set()
    for i in range(len(path1)):
      u, v = path1[i], path1[(i+1)%len(path1)]
      edges1.add((min(u, v), max(u, v)))
      
    max_attempts = 100
    for _ in range(max_attempts):
      path2 = np.random.permutation(self.num_cities)
      disjoint = True
      for i in range(len(path2)):
        u, v = path2[i], path2[(i+1)%len(path2)]
        if (min(u, v), max(u, v)) in edges1:
          disjoint = False
          break
      if disjoint:
        return (path1, path2)
        
    # Fallback if random fails consistently (unlikely for large N)
    return (path1, np.random.permutation(self.num_cities))