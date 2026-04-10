import os
import sys
import time
import numpy as np
import random
from src.problem import SantaProblem
from src.algorithms import hill_climbing, simulated_annealing, genetic_algorithm

def run_experiment(algo_choice, size, num_runs=5):
  seeds = [10, 20, 30, 40, 50]  # Standardized seeds for fair comparison
  algo_names = {'1': 'Hill Climbing', '2': 'Simulated Annealing', '3': 'Genetic Algorithm', '4': 'Random Solution'}
  algo_name = algo_names[algo_choice]

  results = []
  print(f"\n--- Starting {num_runs}-Seed Benchmark for {algo_name} on {size} cities ---")

  for i in range(num_runs):
    seed = seeds[i]
    print(f"Run {i+1}/{num_runs} (Seed: {seed})...", end=" ", flush=True)
    
    np.random.seed(seed)
    random.seed(seed)
    
    try:
      problem = SantaProblem(size)
    except Exception as e:
      print(f"\nError loading dataset: {e}")
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
      print("Failed.")
      continue
      
    is_valid, _ = problem.validate_path(final_path)
    if is_valid:
      final_dist = problem.calculate_distance(final_path)
      improvement = initial_dist - final_dist
      pct_improvement = (improvement / initial_dist) * 100
      
      results.append({
        'final': final_dist,
        'pct': pct_improvement,
        'time': elapsed,
        'path': final_path,
        'problem': problem
      })
      print(f"Done! ({pct_improvement:.2f}% improvement in {elapsed:.2f}s)")
    else:
      print("Invalid path produced.")

  if results:
    avg_final = np.mean([r['final'] for r in results])
    std_final = np.std([r['final'] for r in results])
    avg_pct = np.mean([r['pct'] for r in results])
    avg_time = np.mean([r['time'] for r in results])
    
    print("\n" + "=" * 50)
    print(f"  BENCHMARK SUMMARY - {algo_name} ({size} cities)")
    print("=" * 50)
    print(f"Average Final Distance: {avg_final:.2f} (± {std_final:.2f})")
    print(f"Average Improvement Percentage: {avg_pct:.2f}%")
    print(f"Average Execution Time: {avg_time:.2f}s")
    print("=" * 50)
    
    # Log summary to results.txt
    with open("results.txt", "a") as f:
      f.write(f"\n--- BENCHMARK SUMMARY ({algo_name}, {size} cities, {num_runs} seeds) ---\n")
      f.write(f"Avg Final Distance: {avg_final:.2f} (+- {std_final:.2f})\n")
      f.write(f"Avg Improvement %: {avg_pct:.2f}%\n")
      f.write(f"Avg Execution Time: {avg_time:.2f}s\n")
      f.write("-" * 50 + "\n")
    print(">> Benchmark summary saved to results.txt")

    # Save visualization for the last run (large sizes are adaptively downsampled in plot_solution)
    last_run = results[-1]
    print("\nSaving plot visualization (last run) to 'solution.png'...")
    last_run['problem'].plot_solution(last_run['path'], title=f"{algo_name} Final Run ({size} cities)")

def run_menu():
  while True:
    print("\n" + "=" * 40)
    print("  Kaggle TSP 2012 - Algorithms Menu  ")
    print("=" * 40)
    
    print("\nSelect an Algorithm:")
    print("1. Hill Climbing")
    print("2. Simulated Annealing")
    print("3. Genetic Algorithm")
    print("4. Random Solution (Baseline)")
    print("0. Exit")
    
    algo_choice = input("\nEnter choice (0-4): ").strip()
    
    if algo_choice == '0':
      print("Exiting...")
      sys.exit(0)
      
    if algo_choice not in ['1', '2', '3', '4']:
      print("Invalid algorithm choice. Please try again.")
      continue
      
    print("\nSelect Dataset Size:")
    print("1. 10 cities")
    print("2. 100 cities")
    if algo_choice != '1':
      print("3. 1000 cities")
      if algo_choice == '3':
        print("4. 10000 cities")
        print("5. 150000 cities (slower)")
      else:
        print("4. 10000 cities")
        print("5. 150000 cities")
    print("0. Back to algorithms")
    
    size_map = {'1': '10', '2': '100'}
    if algo_choice != '1':
      size_map.update({'3': '1000', '4': '10000', '5': '150000'})
      prompt = "\nEnter choice (0-5): "
    else:
      prompt = "\nEnter choice (0-2): "

    size_choice = input(prompt).strip()
    
    if size_choice == '0':
      continue
      
    if size_choice not in size_map:
      print("Invalid size choice. Please try again.")
      continue
      
    size = size_map[size_choice]
    
    # Run the multi-seed experiment
    run_experiment(algo_choice, size)
    
    input("\nPress Enter to return to the main menu...")

if __name__ == "__main__":
  run_menu()
