import numpy as np

def apply_2opt(path):
    """reverses a random segment to resolve crossings"""
    if len(path) < 3:
        return path.copy()
    new_path = path.copy()
    idx1, idx2 = sorted(np.random.choice(len(path), 2, replace=False))
    new_path[idx1:idx2] = new_path[idx1:idx2][::-1]
    return new_path

def apply_inter_swap(path1, path2):
    """swaps a random city between the two paths to balance them"""
    new_path1, new_path2 = path1.copy(), path2.copy()
    if len(path1) > 0 and len(path2) > 0:
        idx1 = np.random.randint(len(path1))
        idx2 = np.random.randint(len(path2))
        new_path1[idx1], new_path2[idx2] = new_path2[idx2], new_path1[idx1]
    return new_path1, new_path2

def get_neighbors(problem, current_state):
    """generate 2-opt and inter-path neighbors"""
    neighbors = []
    path1, path2 = current_state
    
    # generate 50 random neighbors to keep it fast
    for _ in range(50):
        # randomly choose an operator
        op = np.random.choice(['2opt1', '2opt2', 'inter'])
        
        if op == '2opt1':
            new_path1 = apply_2opt(path1)
            new_path2 = path2.copy()
        elif op == '2opt2':
            new_path1 = path1.copy()
            new_path2 = apply_2opt(path2)
        else:
            new_path1, new_path2 = apply_inter_swap(path1, path2)
            
        new_state = (new_path1, new_path2)
        
        # only keep valid paths
        is_valid, _ = problem.validate_path(new_state)
        if is_valid:
            neighbors.append(new_state)
            
    return neighbors


def hill_climbing(problem, initial_state=None):
    # initial state
    if initial_state is None:
        current = problem.get_random_solution()
    else:
        current = initial_state
    
    while True:
        # get neighbors
        neighbors = get_neighbors(problem, current)
        
        if not neighbors:
            return current
            
        best_neighbor = None
        best_value = float('inf') # look for the lowest distance
        
        # evaluate neighbors
        for neighbor in neighbors:
            val = problem.calculate_distance(neighbor)
            if val < best_value:
                best_value = val
                best_neighbor = neighbor
                
        current_value = problem.calculate_distance(current)
        
        # return if no improvement (if the lowest neighbor is still worse than current)
        if best_value >= current_value:
            return current
            
        current = best_neighbor
