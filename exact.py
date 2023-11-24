# Author: Julia Kaeppel
import math

# Exact brute-force solution to the TSP.
#
# cities: Set of all cities to be traveled between.
# Returns a traversal order. 
def tsp_exact(cities: set[tuple[float, float]]) -> list[tuple[float, float]]:
    return list(reversed(__tsp_recursive(cities, None)[0]))

# Recursive TSP implementation.
#
# unvisited: Set of currently unvisited cities.
# entering: The first city that is chosen (that must be returned to).
# Returns the shortest path and its cost.
def __tsp_recursive(unvisited: set[tuple[float, float]], entering: tuple[float, float]) -> tuple[list[tuple[float, float]], float]:
    # Handle base case
    if not unvisited:
        return [], 0

    shortest_path = None
    min_cost = float("inf")

    # Test each path
    for city in unvisited:
        # Remove city from unvisited
        unvisited.remove(city)

        # Perform recursion
        if entering:
            path, cost = __tsp_recursive(unvisited, entering)
        else:
            # Set entering
            path, cost = __tsp_recursive(unvisited, city)

        # Update cost
        if path:
            cost += __dist(city, path[-1])
        else:
            cost += __dist(city, entering)

        # Update shortest path
        if cost < min_cost:
            # Only update path when a new shortest path is found
            path.append(city)
            shortest_path = path
            min_cost = cost

        # Add city back to unvisited
        unvisited.add(city)

    return shortest_path, min_cost

# Calculates the distance between two cities
def __dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
