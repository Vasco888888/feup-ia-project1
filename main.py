import os
import sys
from src.problem import SantaProblem
from src.algorithms import hill_climbing, simulated_annealing, genetic_algorithm

def run_menu():
  while True:
    print("\n" + "=" * 40)
    print("  Kaggle TSP 2012 - Algorithms Menu  ")
    print("=" * 40)
    
    print("\nSelect an Algorithm:")
    print("1. Hill Climbing")
    print("2. Simulated Annealing")
    print("3. Genetic Algorithm")
    print("0. Exit")
    
    algo_choice = input("\nEnter choice (0-3): ").strip()
    
    if algo_choice == '0':
      print("Exiting...")
      sys.exit(0)
      
    if algo_choice not in ['1', '2', '3']:
      print("Invalid algorithm choice. Please try again.")
      continue
      
    print("\nSelect Dataset Size:")
    print("1. 10 cities")
    print("2. 100 cities")
    print("3. 1000 cities")
    print("4. 10000 cities")
    print("5. 150000 cities")
    print("0. Back to algorithms")
    
    size_map = {'1': '10', '2': '100', '3': '1000', '4': '10000', '5': '150000'}
    size_choice = input("\nEnter choice (0-5): ").strip()
    
    if size_choice == '0':
      continue
      
    if size_choice not in size_map:
      print("Invalid size choice. Please try again.")
      continue
      
    size = size_map[size_choice]
    
    print(f"\n--- Loading {size} cities ---")
    try:
      problem = SantaProblem(size)
    except Exception as e:
      print(f"Error loading dataset: {e}")
      continue
      
    # Prepare initial solution
    initial_state = problem.get_random_solution()
    initial_val = problem.calculate_distance(initial_state)
    print(f"Initial Random Distance: {initial_val:.2f}")

    final_path = None
    
    # run selected algorithm
    if algo_choice == '1':
      print("\n--- Running Hill Climbing ---")
      final_path = hill_climbing(problem, initial_state)
    
    elif algo_choice == '2':
      print("\n--- Running Simulated Annealing ---")
      final_path = simulated_annealing(problem, initial_state)
      
    elif algo_choice == '3':
      print("\n--- Running Genetic Algorithm ---")
      final_path = genetic_algorithm(problem, initial_state)
      
    if final_path is None:
      print("No path returned from algorithm. Returning to menu.")
      continue
      
    # validate and print results
    print("\n--- Results ---")
    is_valid, message = problem.validate_path(final_path)
    print(f"Path Validation: {message}")

    if is_valid:
      final_distance = problem.calculate_distance(final_path)
      print(f"Final Optimized Distance: {final_distance:.2f}")
      print(f"Total Improvement: {initial_val - final_distance:.2f}")

      # only plot if <= 1000 cities or matplotlib freezes
      if problem.num_cities <= 1000:
        print("\nSaving plot visualization to 'solution.png'...")
        algo_names = {'1': 'Hill Climbing', '2': 'Simulated Annealing', '3': 'Genetic Algorithm'}
        problem.plot_solution(final_path, title=f"{algo_names[algo_choice]} Solution ({size} cities)")

    # wait before menu
    input("\nPress Enter to return to the main menu...")

if __name__ == "__main__":
  run_menu()