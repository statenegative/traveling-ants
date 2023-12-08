# This program implements the "ant-cycle" ant colony optimization algorithm
# described by Colorni, et al. in "Distributed Optimization by Ant Colonies".
# This algorithm can be used to heuristically solve the traveling salesman problem,
# and with small modifications can solve other related shortest-path algorithms.
#
# Author: Julia Kaeppel

import math
import numpy as np
import random
import typing

# A City is a tuple representing an (x, y) coordinate.
City = tuple[float, float]

# Heuristically solves the traveling salesman problem using the ant-cycle ant
# colony optimization algorithm.
#
# cities: A set of cities to find the shortest path between.
# alpha: The relative weight of pheromone strengths.
# beta: The relative weight of distances.
# rho: The evaporation coefficient of pheromones.
# Q_3: The weight of pheromones deposited by ants.
# cycles: The number of iterations to run.
def tsp_aco(cities: set(City), alpha: float, beta: float, rho: float, Q_3: float, cycles: int) -> list[City]:
    # Number cities
    cities = dict(enumerate(cities))
    n = len(cities)
    # Initialize tau matrix with values of 1.0
    tau = np.full((n, n), 1.0)

    # Keep track of shortest path
    shortest_dist = np.inf

    # Run each cycle of the simulation
    for _ in range(cycles):
        # Initialize next tau matrix with pheromone evaporation
        tau_next = tau * rho

        # Start at each city
        for i in range(n):
            unvisited = cities.copy()
            path = []

            # Iterate until all cities have been visited
            while len(unvisited) > 1:
                # Select edge, add it to the path, remove city from unvisited, and move ant
                j = choose_edge(unvisited, tau, alpha, beta, i)
                path.append((i, j))
                del unvisited[i]
                i = j
            
            # Update new tau matrix
            L = path_len(cities, path)
            for edge in path:
                tau_next[edge[0], edge[1]] += Q_3 / L

            # Update shortest path if one was found
            if L < shortest_dist:
                shortest_path = path
                shortest_dist = L
    
    return [cities[shortest_path[0][0]]] + [cities[edge[1]] for edge in shortest_path]

def path_len(cities: dict[int, City], path: list[tuple[int, int]]) -> float:
    # Initialize length to the distance to return to the starting city
    length = dist(cities[path[0][0]], cities[path[-1][1]])
    # Add length of each edge to total length
    for edge in path:
        length += dist(cities[edge[0]], cities[edge[1]])

    return length

# Randomly selects an edge from city i.
#
# cities: A map of unvisited cities (including city i).
# tau: The pheromone weights for each edge.
# alpha: The relative weight of pheromone strengths.
# beta: The relative weight of distances.
# i: The index of the origin city of the edge.
# Returns the index of the destination city of the edge.
def choose_edge(cities: dict[int, City], tau: np.ndarray, alpha: float, beta: float, i: int) -> int:
    # Select a random number in the range 0.0 <= r < 1.0
    r = random.random()
    # Calculate sum of weights
    ws = weight_sum(cities, tau, alpha, beta, i)

    # Keep track of current accumulated probability
    prob_sum = 0.0
    for j in cities.keys():
        # Skip edge from city to itself
        if j == i:
            continue

        # Increment accumulated probability until random value is selected
        p = prob(cities, ws, tau, alpha, beta, i, j)
        prob_sum += p
        if prob_sum >= r:
            break
    return j

# Calculates the weighted probability of selecting the edge from city i to city j.
#
# cities: A map of unvisited cities.
# tau: The pheromone weights for each edge.
# alpha: The relative weight of pheromone strengths.
# beta: The relative weight of distances.
# i: The index of the origin city of the edge.
# j: The index of the destination city of the edge.
# Returns the probability of the edge being selected.
def prob(cities: dict[int, City], weight_sum: float, tau: np.ndarray, alpha: float, beta: float, i: int, j: int) -> float:
    return weight(cities, tau, alpha, beta, i, j) / weight_sum

# Calculates the sum of weights of the edges between one city and all other unvisited cities.
#
# cities: A map of unvisited cities.
# tau: The pheromone weights for each edge.
# alpha: The relative weight of pheromone strengths.
# beta: The relative weight of distances.
# i: The index of the origin city.
# Returns the sum of weights.
def weight_sum(cities: dict[int, City], tau: np.ndarray, alpha: float, beta: float, i: int) -> float:
    return sum([weight(cities, tau, alpha, beta, i, j) for j in cities.keys() if j != i])
    

# Calculates the weight of the edge from city i to city j.
#
# cities: A map of unvisited cities.
# tau: The pheromone weights for each edge.
# alpha: The relative weight of pheromone strengths.
# beta: The relative weight of distances.
# i: The index of the origin city of the edge.
# j: The index of the destination city of the edge.
# Returns the weight of the specified edge.
def weight(cities: dict[int, City], tau: np.ndarray, alpha: float, beta: float, i: int, j: int) -> float:
    return (tau[i, j] ** alpha) * (eta(cities, i, j) ** beta)

# Calculates the visibility between two cities, which is the inverse of their distance.
#
# cities: A map of unvisited cities.
# i: The index of the first city. 
# j: The index of the second city.
# Returns the visibility between the two cities.
def eta(cities: dict[int, City], i: int, j: int) -> float:
    return 1.0 / dist(cities[i], cities[j])

# Calculates the distance between two cities.
#
# a: The first city.
# b: The second city.
# Returns the distance between the two cities.
def dist(a: City, b: City) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def main():
    cities = { (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3) }
    alpha = 1.25
    beta = 4
    rho = 0.7
    Q_3 = 100
    cycles = 100
    print(tsp_aco(cities, alpha, beta, rho, Q_3, cycles))

if __name__ == "__main__":
    main()
