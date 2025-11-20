# EVOLVE-BLOCK-START
"""Multi-Objective Capacitated Vehicle Routing Problem (MOCVRP) evaluator"""
import numpy as np
import random
import os
import math

def load_problem_data(problem_id):
    # 使用项目内的相对路径
    MCCE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(MCCE_ROOT, "data", "problems", "MOCVRP", f"MOCVRP_200_{problem_id}.txt")
    customers = []
    with open(data_path, 'r') as f:
        lines = f.readlines()

    # 第一行：车辆数量和每辆车容量
    vehicle_info = lines[0].strip().split(',')
    n_vehicles = int(vehicle_info[0])
    vehicle_capacity = float(vehicle_info[1])

    # 第二行：仓库坐标
    depot_coords = list(map(float, lines[1].strip().split(',')))

    # 第三行：客户数量
    n_customers = int(lines[2].strip())

    # 从第四行开始读取客户数据 (x, y, demand)
    for line in lines[3:3+n_customers]:
        line = line.strip()
        if line:
            x, y, demand = map(float, line.split(','))
            customers.append([x, y, demand])

    return n_vehicles, vehicle_capacity, depot_coords, np.array(customers)

def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two coordinates"""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def calculate_vrp_metrics(solution, n_vehicles, vehicle_capacity, depot_coords, customers):
    """
    Calculate the total distance and makespan for a VRP solution.

    Args:
        solution: List of routes, each route is a list of customer indices
        n_vehicles: Number of vehicles
        vehicle_capacity: Capacity of each vehicle
        depot_coords: Depot coordinates [x, y]
        customers: Array of shape (n_customers, 3) with [x, y, demand] for each customer

    Returns:
        tuple: (total_distance, makespan, is_feasible, error_message)
    """
    total_distance = 0
    max_route_distance = 0
    all_customers_served = set(range(len(customers)))

    # Check if all customers are assigned
    served_customers = set()
    for route in solution:
        served_customers.update(route)

    if served_customers != all_customers_served:
        missing = all_customers_served - served_customers
        extra = served_customers - all_customers_served
        return 0, 0, False, f"Customers not properly assigned. Missing: {missing}, Extra: {extra}"

    # Calculate distance and capacity for each route
    for i, route in enumerate(solution):
        if len(route) == 0:
            continue

        route_distance = 0
        route_capacity = 0

        # From depot to first customer
        current_pos = depot_coords
        for customer_idx in route:
            if customer_idx >= len(customers):
                return 0, 0, False, f"Invalid customer index {customer_idx}"
            customer_coords = customers[customer_idx][:2]
            route_distance += calculate_distance(current_pos, customer_coords)
            route_capacity += customers[customer_idx][2]
            current_pos = customer_coords

        # Back to depot
        route_distance += calculate_distance(current_pos, depot_coords)
        total_distance += route_distance
        max_route_distance = max(max_route_distance, route_distance)

        # Check capacity constraint
        if route_capacity > vehicle_capacity:
            return 0, 0, False, f"Route {i} capacity {route_capacity:.2f} exceeds vehicle capacity {vehicle_capacity:.2f}"

    return total_distance, max_route_distance, True, ""

def repair_solution(solution, n_vehicles, vehicle_capacity, customers):
    """
    Repair an infeasible solution by reassigning customers to satisfy capacity constraints.
    Uses a simple greedy approach.
    """
    if not solution:
        return []

    # Flatten all routes and sort customers by demand (largest first)
    all_customers = []
    for route in solution:
        all_customers.extend(route)

    if not all_customers:
        return [[] for _ in range(n_vehicles)]

    # Sort customers by demand (greedy assignment)
    customer_demands = [(idx, customers[idx][2]) for idx in all_customers]
    customer_demands.sort(key=lambda x: x[1], reverse=True)

    # Create new routes
    new_routes = [[] for _ in range(n_vehicles)]
    route_capacities = [0] * n_vehicles

    for customer_idx, demand in customer_demands:
        # Find the route with most remaining capacity
        best_route = max(range(n_vehicles),
                        key=lambda i: vehicle_capacity - route_capacities[i])

        if route_capacities[best_route] + demand <= vehicle_capacity:
            new_routes[best_route].append(customer_idx)
            route_capacities[best_route] += demand
        else:
            # If no route can take this customer, assign to route with least capacity usage
            worst_route = min(range(n_vehicles),
                            key=lambda i: route_capacities[i])
            new_routes[worst_route].append(customer_idx)
            route_capacities[worst_route] += demand

    return new_routes

def generate_initial_population(config, seed=42):
    num_samples = 50
    problem_id = config.get('problem_id', 1)
    n_vehicles, vehicle_capacity, depot_coords, customers = load_problem_data(problem_id)
    n_customers = len(customers)

    np.random.seed(seed)
    samples = []

    for _ in range(num_samples):
        # Create random routes
        customer_indices = list(range(n_customers))
        np.random.shuffle(customer_indices)

        # Distribute customers among vehicles
        routes = [[] for _ in range(n_vehicles)]
        route_capacities = [0] * n_vehicles

        for customer_idx in customer_indices:
            demand = customers[customer_idx][2]
            # Find a random route that can accommodate this customer
            feasible_routes = []
            for i in range(n_vehicles):
                if route_capacities[i] + demand <= vehicle_capacity:
                    feasible_routes.append(i)

            if feasible_routes:
                route_idx = np.random.choice(feasible_routes)
            else:
                # If no route can take it, assign to route with most remaining capacity
                route_idx = max(range(n_vehicles),
                              key=lambda i: vehicle_capacity - route_capacities[i])

            routes[route_idx].append(customer_idx)
            route_capacities[route_idx] += demand

        # Repair solution to ensure feasibility
        routes = repair_solution(routes, n_vehicles, vehicle_capacity, customers)

        # Convert solution to string format
        solution_str = convert_solution_to_str(routes)
        samples.append(solution_str)

    return samples

def convert_solution_to_str(solution):
    """Convert solution (list of routes) to string format"""
    import json
    # Convert solution to a simple list of lists format
    solution_str = json.dumps(solution)
    final_string = f"solution = {solution_str}"
    return final_string

def validate_solution(solution, n_customers):
    """Validate that solution is a valid VRP solution"""
    if not isinstance(solution, list):
        return False, "Solution must be a list of routes"

    all_customers = set()
    for route in solution:
        if not isinstance(route, list):
            return False, "Each route must be a list of customer indices"
        for customer_idx in route:
            if not isinstance(customer_idx, int):
                return False, "Customer indices must be integers"
            if customer_idx in all_customers:
                return False, f"Customer {customer_idx} assigned to multiple routes"
            if customer_idx < 0 or customer_idx >= n_customers:
                return False, f"Customer index {customer_idx} out of range [0, {n_customers-1}]"
            all_customers.add(customer_idx)

    # Check if all customers are assigned
    if len(all_customers) != n_customers:
        return False, f"Not all customers assigned. Expected {n_customers}, got {len(all_customers)}"

    return True, ""

def validate_solution_with_constraints(solution, n_vehicles, vehicle_capacity, depot_coords, customers):
    """
    Validate solution including both format and constraint checks.

    Args:
        solution: List of routes (each route is list of customer indices)
        n_vehicles: Number of vehicles
        vehicle_capacity: Capacity of each vehicle
        depot_coords: Depot coordinates
        customers: Customer data

    Returns:
        tuple: (is_valid_format, is_feasible, total_distance, error_message)
    """
    # Check format validity
    is_valid_format, error_msg = validate_solution(solution, len(customers))
    if not is_valid_format:
        return False, False, 0, error_msg

    # Check number of routes
    if len(solution) != n_vehicles:
        return False, False, 0, f"Expected {n_vehicles} routes, got {len(solution)}"

    # Calculate VRP metrics
    total_distance, makespan, is_feasible, error_msg = calculate_vrp_metrics(
        solution, n_vehicles, vehicle_capacity, depot_coords, customers)

    return True, is_feasible, total_distance, error_msg

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def calculate_route_distance(route, depot, customers):
    if not route:
        return 0.0
    pts = np.vstack((depot, customers[route, :2], depot))
    diffs = pts[1:] - pts[:-1]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

def calculate_route_capacity(route, customers):
    if not route:
        return 0.0
    return float(np.sum(customers[route, 2]))

def two_opt(route, depot, customers):
    n = len(route)
    if n < 4:
        return route

    best_route = route.copy()
    best_dist = calculate_route_distance(best_route, depot, customers)
    improved = True

    while improved:
        improved = False
        max_j = min(n - 1, 15)
        for i in range(1, n - 2):
            for j in range(i + 1, max_j):
                if j - i == 1:
                    continue
                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                new_dist = calculate_route_distance(new_route, depot, customers)
                if new_dist + 1e-15 < best_dist:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    return best_route

def generate_3opt_variations(route, i, j, k):
    a, b, c, d, e, f = i, i + 1, j, j + 1, k, k + 1
    s1 = route[:b]
    s2 = route[b:c]
    s3 = route[c:e]
    s4 = route[e:]
    variants = [
        s1 + s2[::-1] + s3 + s4,
        s1 + s2 + s3[::-1] + s4,
        s1 + s2[::-1] + s3[::-1] + s4,
        s1 + s3 + s2 + s4,
        s1 + s3[::-1] + s2 + s4,
        s1 + s3 + s2[::-1] + s4,
        s1 + s3[::-1] + s2[::-1] + s4,
    ]
    return [v for v in variants if v != route]

def three_opt(route, depot, customers):
    n = len(route)
    if n < 5:
        return route

    best_route = route.copy()
    best_dist = calculate_route_distance(best_route, depot, customers)
    improved = True
    max_checks = 400
    checks = 0

    while improved and checks < max_checks:
        improved = False
        for i in range(n - 4):
            for j in range(i + 2, n - 2):
                for k in range(j + 2, n):
                    for candidate in generate_3opt_variations(best_route, i, j, k):
                        dist = calculate_route_distance(candidate, depot, customers)
                        checks += 1
                        if dist + 1e-15 < best_dist:
                            best_route = candidate
                            best_dist = dist
                            improved = True
                            break
                    if improved or checks >= max_checks:
                        break
                if improved or checks >= max_checks:
                    break
            if improved or checks >= max_checks:
                break
    return best_route

def intra_route_optimize(routes, depot, customers):
    optimized = []
    for r in routes:
        if len(r) < 4:
            optimized.append(r)
            continue
        r_2opt = two_opt(r, depot, customers)
        if 5 <= len(r_2opt) <= 12:
            r_3opt = three_opt(r_2opt, depot, customers)
            optimized.append(r_3opt)
        else:
            optimized.append(r_2opt)
    return optimized

def calculate_vrp_score(routes, depot, customers):
    if not routes:
        return float("inf"), float("inf")
    dists = np.array([calculate_route_distance(r, depot, customers) for r in routes])
    makespan = np.max(dists) if dists.size > 0 else 0.0
    total_distance = np.sum(dists)
    return float(makespan), float(total_distance)

def dominates(m1, d1, m2, d2):
    # Pareto dominance for minimization: better or equal in all and strictly better in at least one
    return (m1 <= m2 and d1 < d2) or (m1 < m2 and d1 <= d2)

def inter_route_local_search(n_vehicles, vehicle_capacity, depot, customers, routes):
    if not routes:
        return routes

    rng = np.random.default_rng()
    best_routes = [r.copy() for r in routes]
    best_makespan, best_total_distance = calculate_vrp_score(best_routes, depot, customers)

    max_iter = 220  # Increased for better exploration
    iteration = 0
    improved = True

    def cache_capacities(rs):
        return [calculate_route_capacity(r, customers) for r in rs]

    while improved and iteration < max_iter:
        iteration += 1
        improved = False

        route_indices = list(range(len(best_routes)))
        rng.shuffle(route_indices)

        capacities = cache_capacities(best_routes)

        # Inter-route moves: relocate one customer to another route if Pareto improving
        for i_idx in route_indices:
            for j_idx in route_indices:
                if i_idx == j_idx:
                    continue
                route_i = best_routes[i_idx]
                route_j = best_routes[j_idx]
                if not route_i:
                    continue

                cap_i = capacities[i_idx]
                cap_j = capacities[j_idx]

                for pos in range(len(route_i)):
                    cust = route_i[pos]
                    demand_c = customers[cust, 2]

                    cap_i_new = cap_i - demand_c
                    cap_j_new = cap_j + demand_c
                    if cap_j_new <= vehicle_capacity and cap_i_new >= 0:
                        new_route_i = route_i[:pos] + route_i[pos + 1 :]
                        if calculate_route_capacity(new_route_i, customers) <= vehicle_capacity:
                            new_route_j = route_j + [cust]

                            # Optimize affected routes with 2-opt (fast)
                            new_route_i_opt = two_opt(new_route_i, depot, customers) if len(new_route_i) >= 4 else new_route_i
                            new_route_j_opt = two_opt(new_route_j, depot, customers) if len(new_route_j) >= 4 else new_route_j

                            new_routes = best_routes.copy()
                            new_routes[i_idx] = new_route_i_opt
                            new_routes[j_idx] = new_route_j_opt

                            # Remove empty routes if any
                            new_routes = [r for r in new_routes if r]

                            new_makespan, new_total_distance = calculate_vrp_score(new_routes, depot, customers)

                            if dominates(new_makespan, new_total_distance, best_makespan, best_total_distance):
                                best_routes = new_routes
                                best_makespan = new_makespan
                                best_total_distance = new_total_distance
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # Inter-route swaps: exchange customers between two routes if Pareto improving
        for i_idx in route_indices:
            for j_idx in route_indices:
                if i_idx >= j_idx:
                    continue
                route_i = best_routes[i_idx]
                route_j = best_routes[j_idx]
                if not route_i or not route_j:
                    continue

                cap_i = capacities[i_idx]
                cap_j = capacities[j_idx]

                swapped = False
                for pos_i in range(len(route_i)):
                    for pos_j in range(len(route_j)):
                        cust_i = route_i[pos_i]
                        cust_j = route_j[pos_j]

                        demand_i = customers[cust_i, 2]
                        demand_j = customers[cust_j, 2]

                        cap_i_new = cap_i - demand_i + demand_j
                        cap_j_new = cap_j - demand_j + demand_i

                        if cap_i_new <= vehicle_capacity and cap_j_new <= vehicle_capacity:
                            new_route_i = route_i.copy()
                            new_route_j = route_j.copy()
                            new_route_i[pos_i] = cust_j
                            new_route_j[pos_j] = cust_i

                            new_route_i_opt = two_opt(new_route_i, depot, customers) if len(new_route_i) >= 4 else new_route_i
                            new_route_j_opt = two_opt(new_route_j, depot, customers) if len(new_route_j) >= 4 else new_route_j

                            new_routes = best_routes.copy()
                            new_routes[i_idx] = new_route_i_opt
                            new_routes[j_idx] = new_route_j_opt

                            new_makespan, new_total_distance = calculate_vrp_score(new_routes, depot, customers)

                            if dominates(new_makespan, new_total_distance, best_makespan, best_total_distance):
                                best_routes = new_routes
                                best_makespan = new_makespan
                                best_total_distance = new_total_distance
                                improved = True
                                swapped = True
                                break
                    if swapped:
                        break
                if improved:
                    break
            if improved:
                break

    return best_routes

def optimize_solution(solution, n_vehicles, vehicle_capacity, depot_coords, customers):
    depot = np.array(depot_coords)
    # Repair if necessary
    solution = repair_solution(solution, n_vehicles, vehicle_capacity, customers)
    # Intra-route optimization
    solution = intra_route_optimize(solution, depot, customers)
    # Inter-route local search
    optimized_solution = inter_route_local_search(n_vehicles, vehicle_capacity, depot, customers, solution)
    return optimized_solution

class RewardingSystem:
    def __init__(self, config=None):
        self.config = config
        self.problem_id = config.get('problem_id', 1) if config else 1
        self.n_vehicles, self.vehicle_capacity, self.depot_coords, self.customers = load_problem_data(self.problem_id)
        self.n_customers = len(self.customers)

        # Get optimization directions from config
        self.objs = config.get('goals') if config else ['total_distance', 'makespan']
        self.obj_directions = {obj: config.get('optimization_direction')[i] for i, obj in enumerate(self.objs)}

    def evaluate(self, items, mol_buffer=None):
        valid_items = []
        log_dict = {}

        for item in items:
            scope = {}
            results_dict = {}
            try:
                # Execute the solution string to get solution array
                exec(item.value, {"np": np}, scope)
                solution = scope["solution"]

                # Comprehensive validation
                is_valid_format, is_feasible, original_distance, error_msg = validate_solution_with_constraints(
                    solution, self.n_vehicles, self.vehicle_capacity, self.depot_coords, self.customers)

                if not is_valid_format:
                    print(f"Invalid solution format: {error_msg}")
                    continue

                # Optimize the solution (includes repair if needed)
                optimized_solution = optimize_solution(solution, self.n_vehicles, self.vehicle_capacity, self.depot_coords, self.customers)

                # Calculate metrics for the optimized solution
                total_distance, makespan, is_feasible, error_msg = calculate_vrp_metrics(
                    optimized_solution, self.n_vehicles, self.vehicle_capacity, self.depot_coords, self.customers)

                if not is_feasible:
                    print(f"Optimized solution still infeasible: {error_msg}. Skipping...")
                    continue

                # Store results
                results_dict['original_results'] = {
                    'total_distance': total_distance,
                    'makespan': makespan,
                    'is_feasible': is_feasible
                }

                # Transform results following molecules pattern
                # For VRP, we normalize by estimating maximum possible distance
                # Rough estimate: assume a grid layout and calculate upper bound
                max_possible_distance = self.estimate_max_distance()

                # Normalize to [0,1] range
                normalized_distance = total_distance / max_possible_distance if max_possible_distance > 0 else 0
                normalized_makespan = makespan / max_possible_distance if max_possible_distance > 0 else 0

                # Adjust direction based on optimization_direction
                transformed_distance = self.adjust_direction('total_distance', normalized_distance)
                transformed_makespan = self.adjust_direction('makespan', normalized_makespan)

                results_dict['transformed_results'] = {
                    'total_distance': transformed_distance,
                    'makespan': transformed_makespan
                }

                # Overall score: following molecules pattern
                # Start with best score and subtract transformed values
                overall_score = len(self.objs) * 1.0  # best score
                overall_score -= transformed_distance
                overall_score -= transformed_makespan
                results_dict['overall_score'] = overall_score

                # Assign results to item
                item.assign_results(results_dict)
                # Update item.value with the optimized solution
                item.value = convert_solution_to_str(optimized_solution)
                valid_items.append(item)

            except Exception as e:
                print(f'Execution error: {e}')
                continue

        log_dict['invalid_num'] = len(items) - len(valid_items)
        log_dict['repeated_num'] = 0  # Could implement duplicate detection if needed

        return valid_items, log_dict

    def estimate_max_distance(self):
        """Estimate maximum possible distance for normalization"""
        # Calculate the bounding box
        all_coords = [self.depot_coords] + [customer[:2] for customer in self.customers]
        xs = [coord[0] for coord in all_coords]
        ys = [coord[1] for coord in all_coords]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        diagonal = math.sqrt(width**2 + height**2)

        # Rough estimate: assume each vehicle visits half the customers in a zigzag pattern
        max_distance_per_vehicle = diagonal * (self.n_customers // self.n_vehicles + 2)  # +2 for depot visits
        return max_distance_per_vehicle * self.n_vehicles

    def adjust_direction(self, obj, values):
        """Adjust values based on optimization direction"""
        if self.obj_directions[obj] == 'max':
            # transform to minimization to fit the MOO libraries
            return 1 - values
        elif self.obj_directions[obj] == 'min':
            return values
        else:
            raise NotImplementedError(f'{obj} is not defined for optimization direction! Please define it in "optimization_direction" in your yaml config')