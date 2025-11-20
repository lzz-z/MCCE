# EVOLVE-BLOCK-START
"""Multi-Objective Traveling Salesman Problem (MOTSP) evaluator"""
import numpy as np
import random
import os

def load_problem_data(problem_id):
    # 使用项目内的相对路径
    MCCE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(MCCE_ROOT, "data", "problems", "MOTSP", "problem.txt")
    cities = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                x1, y1, x2, y2 = map(float, line.split(','))
                cities.append([x1, y1, x2, y2])
    return np.array(cities)

def calculate_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def calculate_tour_distance(tour, cities):
    total_distance_obj1 = 0
    total_distance_obj2 = 0
    
    for i in range(len(tour)):
        current_city_idx = tour[i]
        next_city_idx = tour[(i + 1) % len(tour)]
        
        # Calculate distance for objective 1 (x1, y1 coordinates)
        current_city_obj1 = cities[current_city_idx][:2]  # x1, y1
        next_city_obj1 = cities[next_city_idx][:2]        # x1, y1
        distance_obj1 = calculate_distance(current_city_obj1, next_city_obj1)
        total_distance_obj1 += distance_obj1
        
        # Calculate distance for objective 2 (x2, y2 coordinates)
        current_city_obj2 = cities[current_city_idx][2:]  # x2, y2
        next_city_obj2 = cities[next_city_idx][2:]        # x2, y2
        distance_obj2 = calculate_distance(current_city_obj2, next_city_obj2)
        total_distance_obj2 += distance_obj2
    
    return total_distance_obj1, total_distance_obj2

def generate_initial_population(config, seed=42):
    num_samples = 50
    problem_id = config.get('problem_id', 1)  # Default to problem_1
    cities = load_problem_data(problem_id)
    n_cities = len(cities)
    
    np.random.seed(seed)
    samples = []
    
    for _ in range(num_samples):
        # Generate random tour (permutation of cities)
        tour = list(range(n_cities))
        np.random.shuffle(tour)
        
        # Convert tour to string format
        tour_str = convert_tour_to_str(tour)
        samples.append(tour_str)
    
    return samples

def convert_tour_to_str(tour):
    tour_str = np.array2string(np.array(tour), separator=', ', precision=0, suppress_small=True, max_line_width=1000)
    final_string = f"tour = np.array({tour_str})"
    return final_string

def validate_tour(tour, n_cities):
    if len(tour) != n_cities:
        return False
    if set(tour) != set(range(n_cities)):
        return False
    return True

class RewardingSystem:
    def __init__(self, config=None):
        self.config = config
        self.problem_id = config.get('problem_id', 1) if config else 1
        self.cities = load_problem_data(self.problem_id)
        self.n_cities = len(self.cities)
        
        # Get optimization directions from config
        self.objs = config.get('goals') if config else ['distance_obj1', 'distance_obj2']
        self.obj_directions = {obj: config.get('optimization_direction')[i] for i, obj in enumerate(self.objs)}
    
    def evaluate(self, items,mol_buffer=None):
        valid_items = []
        log_dict = {}
        
        for item in items:
            scope = {}
            results_dict = {}
            try:
                # Execute the tour string to get tour array
                exec(item.value, {"np": np}, scope)
                tour = scope["tour"]
                
                # Validate tour
                if not validate_tour(tour, self.n_cities):
                    print(f"Invalid tour: {tour}")
                    continue
                
                # Optimize the tour
                optimized_tour, distance_obj1, distance_obj2 = optimize_tour(tour, self.cities)
                
                # Store results
                results_dict['original_results'] = {
                    'distance_obj1': distance_obj1,
                    'distance_obj2': distance_obj2
                }
                
                # Transform results following molecules pattern
                max_expected_distance = self.n_cities * 2.0  # Rough estimate
                
                # Normalize to [0,1] range
                normalized_obj1 = distance_obj1 / max_expected_distance
                normalized_obj2 = distance_obj2 / max_expected_distance
                
                # Adjust direction based on optimization_direction
                transformed_obj1 = self.adjust_direction('distance_obj1', normalized_obj1)
                transformed_obj2 = self.adjust_direction('distance_obj2', normalized_obj2)
                
                results_dict['transformed_results'] = {
                    'distance_obj1': transformed_obj1,
                    'distance_obj2': transformed_obj2
                }
                
                # Overall score: following molecules pattern
                # Start with best score and subtract transformed values
                overall_score = len(self.objs) * 1.0  # best score
                overall_score -= transformed_obj1
                overall_score -= transformed_obj2
                results_dict['overall_score'] = overall_score
                
                # Assign results to item
                item.assign_results(results_dict)
                # Update item.value with optimized tour
                item.value = convert_tour_to_str(optimized_tour)
                valid_items.append(item)
                
            except Exception as e:
                print(f'Execution error: {e}')
                continue
        
        log_dict['invalid_num'] = len(items) - len(valid_items)
        log_dict['repeated_num'] = 0  # Could implement duplicate detection if needed
        
        return valid_items, log_dict

    def adjust_direction(self, obj, values):
        """Adjust values based on optimization direction"""
        if self.obj_directions[obj] == 'max': 
            # transform to minimization to fit the MOO libraries
            return 1 - values
        elif self.obj_directions[obj] == 'min': 
            return values
        else:
            raise NotImplementedError(f'{obj} is not defined for optimization direction! Please define it in "optimization_direction" in your yaml config')


def nearest_neighbor_tour(combined_dist, start):
    n = combined_dist.shape[0]
    unvisited = set(range(n))
    unvisited.discard(start)
    route = [start]
    current = start

    while unvisited:
        candidates = np.array(list(unvisited))
        if candidates.size == 0:
            break
        dists = combined_dist[current, candidates]
        idx = np.argmin(dists)
        next_city = candidates[idx]
        route.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return route


def route_cost(combined_dist, route):
    n = len(route)
    if n < 2:
        return 0.0
    cost = 0.0
    for i in range(n):
        cost += combined_dist[route[i], route[(i + 1) % n]]
    return cost


def two_opt_improvement(combined_dist, route):
    n = len(route)
    if n < 4:
        return route

    best_route = route.copy()
    best_cost = route_cost(combined_dist, best_route)
    improved = True
    max_iter = 1000
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        iteration += 1

        for i in range(n - 1):
            a, b = best_route[i], best_route[(i + 1) % n]
            # j starts from i+2 to avoid adjacent edges and skip wraparound edge when i=0 and j=n-1
            start_j = i + 2
            end_j = n if i > 0 else n - 1
            for j in range(start_j, end_j):
                c, d = best_route[j], best_route[(j + 1) % n]

                gain = (combined_dist[a, b] + combined_dist[c, d]) - (combined_dist[a, c] + combined_dist[b, d])

                if gain > 1e-12:
                    best_route[i + 1:j + 1] = reversed(best_route[i + 1:j + 1])
                    best_cost -= gain
                    improved = True
                    break
            if improved:
                break

    return best_route

def optimize_tour(tour, cities):
    """
    Optimize a tour for the Multi-Objective TSP using heuristic methods.

    Args:
        tour: Initial tour as np.array or list of city indices
        cities: np.array of shape (n, 4) with city coordinates

    Returns:
        Tuple of (optimized_tour, dist_obj1, dist_obj2)
        optimized_tour: np.array of city indices
        dist_obj1: Total distance for first objective
        dist_obj2: Total distance for second objective
    """
    n = len(cities)
    if n == 0:
        return np.array([]), 0.0, 0.0

    rng = np.random.default_rng()

    coords1 = cities[:, :2]
    coords2 = cities[:, 2:4]

    dist1 = np.linalg.norm(coords1[:, None, :] - coords1[None, :, :], axis=2)
    dist2 = np.linalg.norm(coords2[:, None, :] - coords2[None, :, :], axis=2)

    def safe_normalize(dist):
        ptp = np.ptp(dist)
        if ptp == 0:
            return np.zeros_like(dist)
        return (dist - dist.min()) / ptp

    dist1_norm = safe_normalize(dist1)
    dist2_norm = safe_normalize(dist2)
    combined_dist = 0.5 * dist1_norm + 0.5 * dist2_norm

    # Improve the initial tour
    initial_route = list(tour)
    improved_initial = two_opt_improvement(combined_dist, initial_route)
    best_cost = route_cost(combined_dist, improved_initial)
    best_route = improved_initial

    # Multi-start with nearest neighbor
    max_starts = min(15, n)
    starts = rng.choice(n, size=max_starts, replace=False)

    for start in starts:
        route = nearest_neighbor_tour(combined_dist, start)
        route = two_opt_improvement(combined_dist, route)
        cost = route_cost(combined_dist, route)
        if cost < best_cost:
            best_cost = cost
            best_route = route

    # Compute actual objective distances
    dist_obj1, dist_obj2 = calculate_tour_distance(best_route, cities)

    return np.array(best_route), dist_obj1, dist_obj2

