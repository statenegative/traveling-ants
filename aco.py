# Author: Julia Kaeppel
import numpy as np
import random
import typing

City = tuple[float, float]

# Heuristic ant colony optimization solution to the TSP.
# The edge selection and pheromone update procedures are taken from
# https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms.
#
# cities: Set of all cities to be traveled between.
# n_iters: Number of iterations to run the simulation for.
# m: Number of ants in the simulation.
# alpha: Controls the influence of tau_xy, the pheromone strength of edge xy. 0 <= alpha.
# beta: Controls the influence of eta_xy=1/d_xy, where d_xy is the length of edge xy. 1 <= beta.
# rho: The pheromone evaporation coefficient. 0 <= rho <= 1.
# Q: Q is a constant.
# Returns a traversal order. 
def tsp_aco(cities: set[City], n_iters: int=100, m: int=100, alpha: float=1.0, beta: float=1.0, rho: float=0.1, Q: float=1.0) -> list[City]:
    # Convert cities to ordered list
    cities = list(cities)
    # Matrix of pheromone strengths
    tau = np.zeros((len(cities), len(cities)))
    # Set of ants, starting at random cities
    ants = {cities[random.randrange(len(cities))] for _ in range(m)}

    # Perform iterations
    for _ in n_iters:
        pass

# Performs a tour of the graph with one ant.
#
# ant: The starting/ending point of the ant.
# cities: An ordered list of cities.
# tau: A matrix of pheromone strengths, corresponding to cities.
# alpha: Controls the influence of tau_xy.
# beta: Controls the influence of eta_xy.
# Q: Q is a constant.
# tau_next: A matrix of updated pheromone strengths.
# Returns the traversed path and its cost.
def __update_ant(ant: City, cities: list[City], tau: np.ndarray, alpha: float, beta: float, Q: float, tau_next: np.ndarray) -> tuple[list[City], float]:
    city_indices = {city: i for i, city in enumerate(cities)}
    # Create traversed path and set of unvisited cities
    path = [ant]
    unvisited = city_indices
    del unvisited[ant]

    # Local starting city index
    x = city_indices[ant]
    # Cost of this tour
    L = 0.0

    # Traverse the graph
    while unvisited:
        # Compute total weight for city x
        weighted_sum = 0.0
        for z in unvisited.values():
            eta = 1.0 / __dist(cities[x], cities[z])
            weighted_sum = (alpha * tau[x,z]) * (beta * eta)
        
        # Select city from weighted random distribution
        r = random.random()
        prob_sum = 0.0
        for y in unvisited.values():
            d = __dist(cities[x], cities[y])
            eta = 1.0 / d
            prob = (alpha * tau[x,y]) * (beta * eta) / weighted_sum
            # Check whether city has been found
            if r >= prob_sum and r < prob_sum + prob:
                # City y is selected, update cost
                L += d
                break
            prob_sum += prob
        
        # Append chosen city to path
        path.append(cities[y])

        # Update current city
        x = y
    
    # Add cost of returning to starting city
    L += __dist(path[-1], path[0])
    
    # Update tau_next
    for i in range(len(path)):
        x = city_indices[path[i]]
        y = city_indices[path[(i+1) % len(path)]]
        tau_next[x,y] += Q / L
    
    # Return traversed path and its cost
    return path, L

# Calculates the distance between two cities.
def __dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
