
import numpy as np
from scipy.optimize import minimize
import random

def generate_initial_population(config,seed=42):
    num_samples=50
    n=config.get('n_circles')
    np.random.seed(seed)
    samples = []

    for _ in range(num_samples):
        centers = np.zeros((n, 2))
        radii = np.linspace(0.12, 0.05, n)
        # Hex-like grid placement with jitter
        grid_count = int(n * 0.8)  # ~80%放在网格上
        grid_x = int(np.sqrt(grid_count))
        grid_y = int(np.ceil(grid_count / grid_x))
        x_coords = np.linspace(0.15, 0.85, grid_x)
        y_coords = np.linspace(0.15, 0.85, grid_y)
        count = 0
        for i in range(grid_x):
            for j in range(grid_y):
                if count >= grid_count:
                    break
                jitter = np.random.uniform(-0.015, 0.015, size=2)
                centers[count] = [x_coords[i] + 0.05 * (j % 2), y_coords[j]] + jitter
                count += 1

        # Remaining centers: fully random
        while count < n:
            centers[count] = np.random.rand(2) * 0.7 + 0.15
            count += 1

        # Clip to ensure valid within [0, 1]
        centers = np.clip(centers, 0.01, 0.99)
        samples.append(convert2str(centers,radii))

    return samples


def convert2str(centers,radii):
    centers_str = np.array2string(centers, separator=', ', precision=6, suppress_small=True, max_line_width=1000)
    radii_str = np.array2string(radii, separator=', ', precision=6, suppress_small=True, max_line_width=1000)

    # 拼接为完整字符串
    final_string = (
    "centers = np.array(" + centers_str + ")\n\n"
    "radii = np.array(" + radii_str + ")"
    )
    return final_string

class RewardingSystem:
    def __init__(self,config=None):
        self.config = config
        self.n = config.get('n_circles')
    
    def evaluate(self,items):
        valid_items = []
        log_dict = {}
        for item in items:
            scope = {}
            results_dict = {}
            try:
                exec(item.value, {"np": np}, scope)
                centers = scope["centers"]
                radii = scope["radii"]
                centers,radii,sum_radii = optimize_until_valid(centers,radii,self.n)
                results_dict['original_results'] = {'radii':sum_radii}
                results_dict['transformed_results'] = {'radii':1-sum_radii/5}
                results_dict['overall_score'] = sum_radii
                item.assign_results(results_dict)
                item.value = convert2str(centers,radii)
                valid_items.append(item)
            except:
                print('execution error, return 0')
        log_dict['invalid_num'] = len(items) - len(valid_items)
        log_dict['repeated_num'] = 0 # default is 0. If you remove the repeated items, then fill this attribute with the amount.
        items = valid_items
        return items,log_dict


def has_overlap(centers, radii, tolerance=1e-8):
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist + tolerance < (radii[i] + radii[j]):
                return True
    return False


def has_out_of_bounds(centers, radii, eps=1e-8):
    """
    检查是否有圆超出边界（单位正方形）
    """
    for (x, y), r in zip(centers, radii):
        if (x - r < 0 - eps or x + r > 1 + eps or
            y - r < 0 - eps or y + r > 1 + eps):
            return True
    return False

def optimize_until_valid(centers, radii,n, max_attempts=5):
    for attempt in range(max_attempts):
        centers, radii, sum_radii = optimize_radii(centers, radii,n)
        has_conflict = has_overlap(centers, radii)
        has_border = has_out_of_bounds(centers, radii)

        if not has_conflict and not has_border:
            return centers, radii, sum_radii
        print(f"⚠️ 存在非法情况：{'重叠' if has_conflict else ''}{' 越界' if has_border else ''}")

def optimize_radii(centers,radii,n):
    """
    Construct an optimized arrangement of n circles in a unit square
    using mathematical principles and optimization techniques.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    assert len(centers) == n
    assert len(radii) == n
    #seed = 42
    #random.seed(seed)
    #np.random.seed(seed)
    # Objective function: Negative sum of radii (to maximize)
    def objective(x):
        centers = x[: 2 * n].reshape(n, 2)
        radii = x[2 * n :]
        return -np.sum(radii)

    # Constraint: No overlaps and circles stay within the unit square
    def constraint(x):
        centers = x[: 2 * n].reshape(n, 2)
        radii = x[2 * n :]
        # Overlap constraint
        overlap_constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                overlap_constraints.append(dist - (radii[i] + radii[j]))

        # Boundary constraints
        boundary_constraints = []
        for i in range(n):
            boundary_constraints.append(centers[i, 0] - radii[i])  # x >= radius
            boundary_constraints.append(1 - centers[i, 0] - radii[i])  # x <= 1 - radius
            boundary_constraints.append(centers[i, 1] - radii[i])  # y >= radius
            boundary_constraints.append(1 - centers[i, 1] - radii[i])  # y <= 1 - radius

        return np.array(overlap_constraints + boundary_constraints)

    # Initial guess vector
    x0 = np.concatenate([centers.flatten(), radii])

    # Bounds: Circles stay within the unit square and radii are positive
    bounds = [(0, 1)] * (2 * n) + [(0.03, 0.2)] * n  # radii are positive, up to 0.2

    # Constraints dictionary
    constraints = {"type": "ineq", "fun": constraint}

    # Optimization using SLSQP
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 10000, "ftol": 1e-8},
    )

    # Extract optimized centers and radii
    optimized_centers = result.x[: 2 * n].reshape(n, 2)
    optimized_radii = result.x[2 * n :]

    # Ensure radii are not negative (numerical stability)
    optimized_radii = np.maximum(optimized_radii, 0.001)

    # Calculate the sum of radii
    sum_radii = np.sum(optimized_radii)
    return optimized_centers, optimized_radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()
