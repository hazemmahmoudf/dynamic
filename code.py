import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ===============================
# Load data
# ===============================
devices_df = pd.read_csv("devices.csv")
sites_df   = pd.read_csv("candidate_sites.csv")

devices = list(zip(devices_df['x'], devices_df['y']))
candidate_points = list(zip(sites_df['x'], sites_df['y']))
placement_costs  = list(sites_df['placement_cost_usd'])
site_names       = list(sites_df['name'])

max_range = 12.0  # coverage range

# ===============================
# Helper function
# ===============================
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ===============================
# Fitness function
# ===============================
def fitness(chromosome):
    total_cost = 0
    total_latency = 0
    penalty = 0

    for dev in devices:
        min_dist = float('inf')
        covered = False

        for i, placed in enumerate(chromosome):
            if placed:
                d = distance(dev, candidate_points[i])
                if d <= max_range:
                    covered = True
                    min_dist = min(min_dist, d)

        if not covered:
            penalty += 500_000
        else:
            total_latency += min_dist

    total_cost = sum(
        placement_costs[i]
        for i, placed in enumerate(chromosome) if placed
    )

    return total_cost + total_latency + penalty

# ===============================
# Genetic Algorithm
# ===============================
def genetic_algorithm(pop_size=200, generations=500):
    n = len(candidate_points)

    population = [
        [random.randint(0, 1) for _ in range(n)]
        for _ in range(pop_size)
    ]

    initial_solution = population[0].copy()  # 🔹 initial random solution

    best_solution = None
    best_score = float('inf')
    history = []

    for gen in range(generations):
        population.sort(key=fitness)
        current_best = fitness(population[0])
        history.append(current_best)

        if current_best < best_score:
            best_score = current_best
            best_solution = population[0].copy()

        next_gen = population[:20]  # elitism

        while len(next_gen) < pop_size:
            p1, p2 = random.choices(population[:50], k=2)
            point = random.randint(1, n - 2)
            child = p1[:point] + p2[point:]

            if random.random() < 0.2:  # mutation
                idx = random.randint(0, n - 1)
                child[idx] = 1 - child[idx]

            next_gen.append(child)

        population = next_gen

    return initial_solution, best_solution, best_score, history

# ===============================
# Run GA
# ===============================
init_solution, final_solution, score, history = genetic_algorithm()

print("Initial cloudlets:", sum(init_solution))
print("Final cloudlets:", sum(final_solution))
print("Final Cost + Latency:", round(score))

# ===============================
# Plot 1: GA Convergence Curve
# ===============================
plt.figure()
plt.plot(history)
plt.xlabel("Generation")
plt.ylabel("Best Fitness Value")
plt.title("GA Convergence Curve")
plt.grid(True)
plt.show()

# ===============================
# Plot helper
# ===============================
def plot_solution(solution, title):
    selected_indices = [i for i, b in enumerate(solution) if b]

    device_assignment = []
    for dev in devices:
        best_idx = None
        best_dist = float('inf')
        for idx in selected_indices:
            d = distance(dev, candidate_points[idx])
            if d <= max_range and d < best_dist:
                best_dist = d
                best_idx = idx
        device_assignment.append(best_idx)

    plt.scatter(
        devices_df['x'],
        devices_df['y'],
        c=device_assignment,
        cmap='viridis',
        s=80,
        label="Devices"
    )

    for i, idx in enumerate(selected_indices):
        x, y = candidate_points[idx]
        plt.scatter(x, y, marker='s', s=200, edgecolors='black')
        plt.text(x + 0.2, y + 0.2, f"C{i}", fontsize=9, weight='bold')

        circle = Circle((x, y), max_range, fill=False, linestyle='--')
        plt.gca().add_patch(circle)

    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.axis("equal")
    plt.grid(True)

# ===============================
# Plot 2: Initial vs Optimized
# ===============================
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plot_solution(init_solution, "Initial Random Solution")

plt.subplot(1,2,2)
plot_solution(final_solution, "Optimized GA Solution")

plt.show()
