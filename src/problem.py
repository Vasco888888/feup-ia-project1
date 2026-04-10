import numpy as np
import pandas as pd
from src.utils import load_cities

class SantaProblem:
    """
    Representation of the Santa 2012 Dual-Path TSP Problem.
    
    The goal is to find two Hamiltonian cycles (paths that visit every city) 
    such that the paths share no common edges (disjoint) and the length 
    of the longer path is minimized.
    """

    def __init__(self, size):
        """
        Initializes the problem by loading the dataset of a specific size.
        
        Args:
            size (int/str): Number of cities to load (10, 100, 1000, etc.)
        """
        self.size = size
        self.cities_df = load_cities(size)
        
        # Determine column names dynamically based on the CSV format
        x_col = 'X' if 'X' in self.cities_df.columns else 'x'
        y_col = 'Y' if 'Y' in self.cities_df.columns else 'y'
        self.id_col = 'CityId' if 'CityId' in self.cities_df.columns else 'id'
        
        # Store coordinates as a NumPy array for vectorized calculations
        self.coords = self.cities_df[[x_col, y_col]].values
        self.num_cities = len(self.cities_df)

    def _path_distance(self, path):
        """
        Calculates the total Euclidean distance for a single closed cycle.
        
        Args:
            path (np.ndarray): Array of city indices representing the visit order.
            
        Returns:
            float: Total distance including the return to the start.
        """
        ordered_coords = self.coords[path]
        # Vectorized distance between consecutive cities
        diffs = np.diff(ordered_coords, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Close the loop: add distance from the last city back to the first city
        last_to_depot = np.sqrt(np.sum((ordered_coords[-1] - self.coords[path[0]])**2))
        return np.sum(distances) + last_to_depot

    def calculate_distance(self, paths):
        """
        Calculates the objective function: max(length(Path1), length(Path2)).
        
        Args:
            paths (tuple): (path1, path2) tuple.
            
        Returns:
            float: The maximum of the two path distances.
        """
        path1, path2 = paths
        dist1 = self._path_distance(path1)
        dist2 = self._path_distance(path2)
        return max(dist1, dist2)

    def validate_path(self, paths):
        """
        Ensures the solution meets all problem constraints:
        1. Both paths visit all cities exactly once.
        2. Both paths are Hamiltonian cycles (implied by length and unique nodes).
        3. No edge is shared between the two paths.
        
        Args:
            paths (tuple): (path1, path2) tuple.
            
        Returns:
            tuple: (bool, str) representing validity and a descriptive message.
        """
        if len(paths) != 2:
            return False, "Expected a tuple of two paths"
          
        path1, path2 = paths
        
        # Check if every city is visited exactly once in each path
        for p_idx, path in enumerate([path1, path2]):
            if len(path) != self.num_cities:
                return False, f"Wrong path length for Path {p_idx+1}: {len(path)} (expected {self.num_cities})"
            unique_cities = set(path)
            if len(unique_cities) != self.num_cities:
                return False, f"Duplicate or missing cities in Path {p_idx+1}"
            
        # Extract edge sets to check for disjointness
        # Edge is represented as a sorted tuple (min, max) to ignore directionality
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
        Generates a visualization of the two paths and saves it as an image.
        Uses adaptive styling (smaller lines/markers) for larger datasets.
        """
        import matplotlib.pyplot as plt
        
        path1, path2 = paths
        
        # Create loop coordinates for Path 1
        ordered_coords1 = self.coords[path1]
        loop_coords1 = np.vstack([ordered_coords1, self.coords[path1[0]]])
        
        # Create loop coordinates for Path 2
        ordered_coords2 = self.coords[path2]
        loop_coords2 = np.vstack([ordered_coords2, self.coords[path2[0]]])

        # Adapt plot visibility based on data density
        if self.num_cities > 10000:
            scatter_size = 1
            line_width = 0.5
            alpha = 0.4
        else:
            scatter_size = 10
            line_width = 1.2
            alpha = 0.6
        
        plt.figure(figsize=(10, 8))
        plt.plot(loop_coords1[:, 0], loop_coords1[:, 1], 'b-', alpha=alpha, linewidth=line_width, label='Path 1')
        plt.plot(loop_coords2[:, 0], loop_coords2[:, 1], 'g-', alpha=alpha, linewidth=line_width, label='Path 2')
        plt.scatter(self.coords[:, 0], self.coords[:, 1], c='red', s=scatter_size, label='Cities')
        
        dist1 = self._path_distance(path1)
        dist2 = self._path_distance(path2)
        
        plt.title(f"{title}\n(Max Distance: {max(dist1, dist2):.2f})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('solution.png')
        print(f"Plot saved to solution.png")
        plt.close()

    def get_random_solution(self):
        """
        Generates a valid initial starting state for the algorithms.
        Uses random permutations while checking for the disjoint constraint.
        """
        path1 = np.random.permutation(self.num_cities)
        
        if self.num_cities < 5:
            # Mathematical impossibility check (K4 is the smallest graph with 2 disjoint H-cycles)
            return (path1, np.random.permutation(self.num_cities))
          
        edges1 = set()
        for i in range(len(path1)):
            u, v = path1[i], path1[(i+1)%len(path1)]
            edges1.add((min(u, v), max(u, v)))
          
        # Try to find a disjoint path randomly before falling back to more complex strategies
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
            
        # Note: For very large datasets, random collision is extremely unlikely.
        return (path1, np.random.permutation(self.num_cities))