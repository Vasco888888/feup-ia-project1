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
        
        # 1. PRE-CALCULATE PRIMES
        self.prime_indicators = self._get_prime_markers(self.cities_df[self.id_col].values)

    def _get_prime_markers(self, city_ids):
        """Returns a boolean array where True means the CityID is prime."""
        max_id = int(max(city_ids))
        sieve = np.ones(max_id + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for p in range(2, int(max_id**0.5) + 1):
            if sieve[p]:
                sieve[p*p : max_id+1 : p] = False
        
        # Return only the markers for the cities we actually have in this instance
        return set([cid for cid in city_ids if sieve[int(cid)]])

    def calculate_distance(self, path):
        """
        Calculates total distance including the 10% penalty on every 10th step
        if the departing city is NOT prime.
        """
        ordered_coords = self.coords[path]
        
        # Standard Euclidean distances
        diffs = np.diff(ordered_coords, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # 1. APPLY THE SANTA PENALTY
        # Penalty is on the 10th step (index 9, 19, 29...)
        # We check the CityId of the city we are DEPARTING from.
        for i in range(9, len(distances), 10):
            city_id = self.cities_df.iloc[path[i]][self.id_col]
            if city_id not in self.prime_indicators:
                distances[i] *= 1.1
        
        # Add distance from last city back to Depot (City 0)
        last_to_depot = np.sqrt(np.sum((ordered_coords[-1] - self.coords[0])**2))
        
        # Check if the return trip is the 10th step
        next_step_num = len(path)
        if next_step_num % 10 == 0:
            last_city_id = self.cities_df.iloc[path[-1]][self.id_col]
            if last_city_id not in self.prime_indicators:
                last_to_depot *= 1.1

        return np.sum(distances) + last_to_depot

    def validate_path(self, path):
        """
        Checks if a path is valid for Santa's TSP.
        """
        if len(path) != self.num_cities:
            return False, f"Wrong path length: {len(path)} (expected {self.num_cities})"
        
        if path[0] != 0:
            return False, "Path must start at City 0 (Depot)"
            
        unique_cities = set(path)
        if len(unique_cities) != self.num_cities:
            return False, "Duplicate or missing cities in path"
            
        return True, "Valid Santa Path"

    def plot_solution(self, path, title="Santa TSP Solution"):
        """
        Plots the cities and the path connecting them.
        """
        import matplotlib.pyplot as plt
        
        ordered_coords = self.coords[path]
        # Append the Depot (City 0) to the end to close the loop
        loop_coords = np.vstack([ordered_coords, self.coords[0]])
        
        plt.figure(figsize=(10, 8))
        plt.plot(loop_coords[:, 0], loop_coords[:, 1], 'b-', alpha=0.6, label='Path')
        plt.scatter(self.coords[:, 0], self.coords[:, 1], c='red', s=10, label='Cities')
        
        # Highlight Depot
        plt.scatter(self.coords[0, 0], self.coords[0, 1], c='green', marker='D', s=100, label='Depot (City 0)')
        
        # Mark Prime Cities
        prime_coords = self.coords[self.cities_df[self.id_col].isin(self.prime_indicators)]
        plt.scatter(prime_coords[:, 0], prime_coords[:, 1], c='gold', s=5, alpha=0.5, label='Primes')
        
        plt.title(f"{title}\n(Total Distance: {self.calculate_distance(path):.2f})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.savefig('solution.png')
        print(f"Plot saved to solution.png")
        plt.close()

    def get_random_solution(self):
        """
        Generates a random path that ALWAYS starts at City 0.
        """
        # Keep 0 at the start, shuffle the rest
        others = np.random.permutation(np.arange(1, self.num_cities))
        return np.insert(others, 0, 0)