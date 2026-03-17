from src.problem import SantaProblem
import os

def test_dataset(size):
    print(f"\n" + "="*40)
    print(f"Testing Dataset Size: {size}")
    print("="*40)
    try:
        # 1. Load the problem
        problem = SantaProblem(size)
        print(f"Loaded {problem.num_cities} cities.")

        # 2. Get a baseline (random) solution
        random_path = problem.get_random_solution()
        print(f"Initial Path: {random_path}")

        # 3. Validate
        is_valid, message = problem.validate_path(random_path)
        print(f"Path Validation: {message}")

        # 4. Calculate Distance
        distance = problem.calculate_distance(random_path)
        print(f"Baseline (Random) Distance: {distance:.2f}")

        # 5. Visualize
        if problem.num_cities == 100:
            plot_path = os.path.join('data', f'solution_{size}.png')
            problem.plot_solution(random_path, title=f"Random Solution ({size} cities)")
            print(f"Plot should be saved as solution.png")

    except Exception as e:
        print(f"Error testing size {size}: {e}")

if __name__ == "__main__":
    # Test with different sizes
    for s in ['10', '100', '1000']:
        test_dataset(s)
