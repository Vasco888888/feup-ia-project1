import time
import numpy as np
import random
import os
import sys

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.problem import SantaProblem
from src.algorithms import hill_climbing, simulated_annealing, genetic_algorithm

def run_benchmark():
  print("\n" + "=" * 50)
  print("  Kaggle TSP 2012 - Multi-Seed Benchmarking Mode  ")
  print("=" * 50)
  
  print("\nSelect an Algorithm:")
  print("1. Hill Climbing")
  print("2. Simulated Annealing")
  print("3. Genetic Algorithm")
  
  algo_choice = input("\nEnter choice (1-3): ").strip()
  if algo_choice not in ['1', '2', '3']:
    print("Invalid choice. Exiting.")
    return

  algo_names = {'1': 'Hill Climbing', '2': 'Simulated Annealing', '3': 'Genetic Algorithm'}
  algo_name = algo_names[algo_choice]

  print("\nSelect Dataset Size:")
  print("1. 10 cities")
  print("2. 100 cities")
  print("3. 1000 cities")
  if algo_choice != '1':
    print("4. 10000 cities")
    print("5. 150000 cities")
  
  size_map = {'1': '10', '2': '100', '3': '1000'}
  if algo_choice != '1':
    size_map.update({'4': '10000', '5': '150000'})
  
  size_choice = input("\nEnter choice: ").strip()
  if size_choice not in size_map:
    print("Invalid size choice.")
    return
  
  size = size_map[size_choice]
  num_runs = 5
  seeds = [10, 20, 30, 40, 50]  # Standardized seeds for fair comparison
  
  results = []
  
  print(f"\n--- Starting benchmark for {algo_name} on {size} cities ({num_runs} runs) ---")
  
  for i, seed in enumerate(seeds):
    print(f"\nRun {i+1}/{num_runs} (Seed: {seed})...")
    
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize problem
    try:
      problem = SantaProblem(size)
    except Exception as e:
      print(f"Error loading dataset: {e}")
      break
      
    initial_state = problem.get_random_solution()
    initial_dist = problem.calculate_distance(initial_state)
    
    start_time = time.time()
    
    if algo_choice == '1':
      final_path = hill_climbing(problem, initial_state)
    elif algo_choice == '2':
      final_path = simulated_annealing(problem, initial_state)
    elif algo_choice == '3':
      final_path = genetic_algorithm(problem, initial_state)
    else:
      final_path = initial_state
      
    elapsed = time.time() - start_time
    
    if final_path is None:
      print("No path returned.")
      continue
      
    is_valid, _ = problem.validate_path(final_path)
    if is_valid:
      final_dist = problem.calculate_distance(final_path)
      improvement = initial_dist - final_dist
      pct_improvement = (improvement / initial_dist) * 100
      
      results.append({
        'initial': initial_dist,
        'final': final_dist,
        'improvement': improvement,
        'pct': pct_improvement,
        'time': elapsed
      })
      print(f"  Result: {final_dist:.2f} | Improvement: {pct_improvement:.2f}% | Time: {elapsed:.2f}s")
    else:
      print("  ERROR: Invalid path produced.")

  if results:
    print("\n" + "=" * 50)
    print(f"  BENCHMARK SUMMARY - {algo_name} ({size} cities)")
    print("=" * 50)
    
    avg_final = np.mean([r['final'] for r in results])
    std_final = np.std([r['final'] for r in results])
    avg_pct = np.mean([r['pct'] for r in results])
    avg_time = np.mean([r['time'] for r in results])
    
    print(f"Average Final Distance: {avg_final:.2f} (± {std_final:.2f})")
    print(f"Average Improvement Percentage: {avg_pct:.2f}%")
    print(f"Average Execution Time: {avg_time:.2f}s")
    print("=" * 50)
    
    # Optional: Log summary to results.txt
    with open("results.txt", "a") as f:
      f.write(f"\n--- BENCHMARK SUMMARY ({algo_name}, {size} cities, {num_runs} seeds) ---\n")
      f.write(f"Avg Final Distance: {avg_final:.2f} (± {std_final:.2f})\n")
      f.write(f"Avg Improvement %: {avg_pct:.2f}%\n")
      f.write(f"Avg Execution Time: {avg_time:.2f}s\n")
      f.write("-" * 50 + "\n")
    print("\nBenchmark summary saved to results.txt!")

if __name__ == "__main__":
  run_benchmark()
