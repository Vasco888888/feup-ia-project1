import os
import sys
import time
from src.problem import SantaProblem

def run_menu():
    while True:
        print("\n" + "=" * 40)
        print("  Kaggle TSP 2012 - Algorithms Menu  ")
        print("=" * 40)
        
        print("\nSelect an Algorithm:")
        print("1. Hill Climbing")
        print("2. Simulated Annealing (Pending Team Member)")
        print("3. Genetic Algorithm (Pending Team Member)")
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

        # Hill Climbing is O(n^2) per iteration — warn for large datasets
        if algo_choice == '1' and size_choice in ('4', '5'):
            print(f"\n[WARNING] Hill Climbing is not practical for {size} cities.")
            print("It scans all O(n²) 2-opt neighbors each iteration and will be extremely slow.")
            confirm = input("Continue anyway? (y/N): ").strip().lower()
            if confirm != 'y':
                continue

        print(f"\n--- Loading {size} cities ---")
        try:
            problem = SantaProblem(size)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            continue

        # ── Algorithm dispatch ─────────────────────────────────────────────
        if algo_choice == '1':
            from src.algorithms.hill_climbing import hill_climbing

            print("\n--- Running Hill Climbing ---")

            initial_state = problem.get_random_solution()

            start_time = time.time()
            final_path = hill_climbing(problem, initial_state)
            elapsed = time.time() - start_time

            final_distance = problem.calculate_distance(final_path)
            print(f"Distance: {final_distance:.2f} | Time: {elapsed:.2f}s")

            algo_name  = "Hill Climbing"
            params_str = "Steepest Ascent (2-opt)"
            the_path_to_validate = final_path
            best_overall_dist    = final_distance
            run_time             = elapsed

        elif algo_choice == '2':
            print("\n--- Running Simulated Annealing ---")
            print("(Pending Team Member)")
            continue

        elif algo_choice == '3':
            print("\n--- Running Genetic Algorithm ---")
            print("(Pending Team Member)")
            continue

        # ── Validation & results ───────────────────────────────────────────
        print("\n--- Results ---")
        is_valid, message = problem.validate_path(the_path_to_validate)
        print(f"Path Validation: {message}")

        if is_valid:
            print(f"Best Distance : {best_overall_dist:.2f}")
            print(f"Execution Time: {run_time:.2f}s")

            log_line = (
                f"{algo_name} | Dataset: {size} | Env: {params_str} "
                f"| Dist: {best_overall_dist:.2f} | Time: {run_time:.2f}s\n"
            )
            try:
                with open("results.txt", "a") as f:
                    f.write(log_line)
                print(">> Saved metrics to results.txt")
            except Exception as e:
                print(f"Failed to log results: {e}")

            if problem.num_cities <= 1000:
                print("\nSaving plot visualization to 'solution.png'...")
                problem.plot_solution(
                    the_path_to_validate,
                    title=f"{algo_name} Solution ({size} cities)"
                )

        input("\nPress Enter to return to the main menu...")

if __name__ == "__main__":
    run_menu()