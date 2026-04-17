# Traveling Santa Problem Solver

This project implements various optimization algorithms to solve the **Santa 2012 Traveling Salesman Problem (TSP)** from Kaggle: [Traveling Santa Problem](https://www.kaggle.com/competitions/traveling-santa-problem/).

The goal is to help Santa deliver toys to 150,000 cities by finding two disjoint Hamiltonian cycles that minimize the maximum length of the two paths.

## Features

- **Hill Climbing**: Local search algorithm for quick improvements.
- **Simulated Annealing**: Stochastic search that can escape local optima using probabilistic moves.
- **Genetic Algorithm**: Population-based evolutionary approach with advanced crossover and mutation.
- **Constraint Handling**: Strictly enforces the disjoint path requirement (no shared edges between the two paths).
- **Scalability**: Optimized to handle datasets from 10 cities up to 150,000 cities.

## Prerequisites

- **Python 3.8+**
- **pip** (Python package installer)

## Installation

1. **Clone the repository** (or navigate to the project folder).
2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Using the Program

The project features a console-based interactive menu for easy benchmarking.

1. **Run the main script**:
   ```bash
   python main.py
   ```
2. **Follow the on-screen prompts**:
   - Select an **Algorithm** (Hill Climbing, Simulated Annealing, or Genetic Algorithm).
   - Select a **Dataset Size** (ranging from 10 cities for debugging to 150,000 for full benchmarking).
   - View progress and results directly in the console.

## Outputs

- **Console Output**: Real-time progress tracking of the optimization.
- **`results.txt`**: A log containing the average improvement percentage, final distances, and execution times for each run.
- **`solution.png`**: A visualization of the final paths produced by the most recent algorithm run.

## Project Structure

- `main.py`: Interactive entry point and benchmarking logic.
- `src/problem.py`: Problem definition, path validation, and distance calculation.
- `src/algorithms/`: Implementation of optimization strategies.
    - `hill_climbing.py`
    - `simulated_annealing.py`
    - `genetic_algorithm.py`
- `data/`: Custom testing datasets (10-10,000 cities) and the original 150,000-city dataset from Kaggle.

## Credits

Developed by **Group 58** for the **Inteligência Artificial (IA)** course at **FEUP** (2025/2026).

- Ana Silva up202308786
- Mário Pereira up202304965
- Vasco Sá up202306731

### Acknowledgments
- **Kaggle**: For the original 150,000-city dataset from the [Traveling Santa Problem](https://www.kaggle.com/competitions/traveling-santa-problem/) competition.
