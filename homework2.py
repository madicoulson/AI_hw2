from collections import deque
import heapq

# Graph including all the cities and their path values
graph = {
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Zerind': {'Oradea': 71, 'Arad': 75},
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Sibiu': {'Oradea': 151, 'Arad': 140, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Pitesti': 97, 'Craiova': 146},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Urziceni': 85, 'Giurgiu': 90},
    'Urziceni': {'Bucharest': 85, 'Vaslui': 142, 'Hirsova': 98},
    'Giurgiu': {'Bucharest': 90},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}

# BFS
def bfs(graph, start_city, goal_city):
    
    # Base case - if the start and goal city are the same, return the goal city
    # as the path and 0 as the cost
    if start_city == goal_city:
        return [goal_city], 0
    
    # Initialize a list to keep track of visited cities
    visited_cities = list()
    
    # Initialize a queue to hold the current city, path to the current city, and the cost
    # This begins with only the starting city, but will be appended to in the traversal
    queue = deque([(start_city, [start_city], 0)]) 

    # Begin the traversal of the queue
    while queue:
        # Pop the current city and add it to the visited_cities list
        current_city, path, cost = queue.popleft()
        visited_cities.append(current_city)

        # Return the path and cost if the current city is equal to the goal city (arrived)
        if current_city == goal_city:
            return path, cost  

        # Check the neighbors of the current city from the graph
        for neighbor, neighbor_cost in graph[current_city].items():
            # If the neighbor is not in visited_cities, the path and cost are incremented by
            # the neighbors path and cost, and each value is appended to the queue
            if neighbor not in visited_cities:
                new_path = path + [neighbor]
                new_cost = cost + neighbor_cost
                queue.append((neighbor, new_path, new_cost))

    # No path found - return empty list and cost of 0
    return [], 0  


# DFS
def dfs(graph, current_city, goal_city, visited=None, path=None):
    
    # Initialize both visited and path if they are None
    if visited is None:
        visited = set()
    if path is None:
        path = []

    # Add both the current_city to both the visited set and path list
    visited.add(current_city)
    path = path + [current_city]

    # Return the path and cost if the current city is equal to the goal city (arrived)
    if current_city == goal_city:
        # Calculate path cost
        path_cost = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
        return path, path_cost

    # Traverse through neighbors of current city based on their edge cost
    for neighbor, _ in sorted(graph[current_city].items(), key=lambda x: x[1]):
        # If the neighbor is not in visited, recursively run dfs again to try to find a new_path
        if neighbor not in visited:
            new_path, path_cost = dfs(graph, neighbor, goal_city, visited, path)
            # If the new_path is found, then it connects to the goal city and is returned
            if new_path:
                return new_path, path_cost

    # No path found - return empty list and cost of 0
    return [], 0 

# A* Search
class Node:
    # Constructor for a Node that represents a city
    def __init__(self, city, g_value):
        self.city = city        # City value
        self.g_value = g_value  # Cost to reach city
        self.f_value = g_value  # Estimated cost of path
        self.parent = None      # Parent Node

def a_star(graph, start_city, goal_city):
    # Initialize node with start_city, empty priority queue open_set, and set for visited cities closed_set
    start_node = Node(start_city, 0)
    open_set = []
    closed_set = set()
    
    # Place start node in priority queue with f_value as the priority
    heapq.heappush(open_set, (start_node.f_value, id(start_node), start_node))
    

    while open_set:
        _, _, current_node = heapq.heappop(open_set)

        # Check if the current city equals the goal city
        if current_node.city == goal_city:
            # Calculate cost and append values to the path
            path = []
            cost = current_node.g_value if current_node else -1  
            while current_node:
                path.append(current_node.city)
                current_node = current_node.parent
            
            # Return the path and the cost
            return path[::-1], cost  

        # Add current city to visited set
        closed_set.add(current_node.city)

        # Check the neighbors of the current city from the graph
        for neighbor, cost in graph[current_node.city].items():
            # If the neighbor is not in the visited closed_set, then calculate a new g_value 
            # for the neighbor (cost), set the parent of the neighbor to the current node, and 
            # push the neighbor node onto the priority queue
            if neighbor not in closed_set:
                g_value = current_node.g_value + cost
                neighbor_node = Node(neighbor, g_value)
                neighbor_node.parent = current_node

                heapq.heappush(open_set, (g_value, id(neighbor_node), neighbor_node))
    
    # No path found - return empty list and cost of 0
    return [], 0  

# Printing the paths for each of the start cities to Bucharest
goal_city = 'Bucharest'

# Test One - from Timisoara
start_city1 = 'Timisoara'
goal_city_dfs = 'Bucharest'
path_dfs, cost_dfs = dfs(graph, start_city1, goal_city)
path_bfs, cost_bfs = bfs(graph, start_city1, goal_city)
path_a, cost_a = a_star(graph, start_city1, goal_city)
print(f"DFS Path from {start_city1} to {goal_city}: {path_dfs}, Cost: {cost_dfs}")
print(f"BFS Path from {start_city1} to {goal_city}: {path_bfs}, Cost: {cost_bfs}")
print(f"A* Path from {start_city1} to {goal_city}: {path_a}, Cost: {cost_a}")

# Test Two - from Oradea
start_city2 = 'Oradea'
path_dfs, cost_dfs = dfs(graph, start_city2, goal_city)
path_bfs, cost_bfs = bfs(graph, start_city2, goal_city)
path_a, cost_a = a_star(graph, start_city2, goal_city)
print(f"DFS Path from {start_city2} to {goal_city}: {path_dfs}, Cost: {cost_dfs}")
print(f"BFS Path from {start_city2} to {goal_city}: {path_bfs}, Cost: {cost_bfs}")
print(f"A* Path from {start_city2} to {goal_city}: {path_a}, Cost: {cost_a}")


# Test Three - from Neamt
start_city3 = 'Neamt'
path_dfs, cost_dfs = dfs(graph, start_city3, goal_city)
path_bfs, cost_bfs = bfs(graph, start_city3, goal_city)
path_a, cost_a = a_star(graph, start_city3, goal_city)
print(f"DFS Path from {start_city3} to {goal_city}: {path_dfs}, Cost: {cost_dfs}")
print(f"BFS Path from {start_city3} to {goal_city}: {path_bfs}, Cost: {cost_bfs}")
print(f"A* Path from {start_city3} to {goal_city}: {path_a}, Cost: {cost_a}")



# -- Paths generated by each algorithm (testing) --

### DFS Paths 
# DFS Path from Timisoara to Bucharest: ['Timisoara', 'Lugoj', 'Mehadia', 'Drobeta', 'Craiova', 'Pitesti', 'Rimnicu Vilcea', 'Sibiu', 'Fagaras', 'Bucharest']
# DFS Path from Oradea to Bucharest: ['Oradea', 'Zerind', 'Arad', 'Timisoara', 'Lugoj', 'Mehadia', 'Drobeta', 'Craiova', 'Pitesti', 'Rimnicu Vilcea', 'Sibiu', 'Fagaras', 'Bucharest']
# DFS Path from Neamt to Bucharest: ['Neamt', 'Iasi', 'Vaslui', 'Urziceni', 'Bucharest']

### BFS Paths
# BFS Path from Timisoara to Bucharest: ['Timisoara', 'Arad', 'Sibiu', 'Fagaras', 'Bucharest'], Cost: 568
# BFS Path from Oradea to Bucharest: ['Oradea', 'Sibiu', 'Fagaras', 'Bucharest'], Cost: 461
# BFS Path from Neamt to Bucharest: ['Neamt', 'Iasi', 'Vaslui', 'Urziceni', 'Bucharest'], Cost: 406

#A* Paths
# A* Path from Timisoara to Bucharest: ['Timisoara', 'Arad', 'Sibiu', 'Rimnicu Vilcea', 'Pitesti', 'Bucharest'], Cost: 536
# A* Path from Oradea to Bucharest: ['Oradea', 'Sibiu', 'Rimnicu Vilcea', 'Pitesti', 'Bucharest'], Cost: 429
# A* Path from Neamt to Bucharest: ['Neamt', 'Iasi', 'Vaslui', 'Urziceni', 'Bucharest'], Cost: 406

# -- Correctness Discussion for each algorithm --
### DFS
# For DFS the algorithm does work correctly and will actively return an empty list if no such path exists between the start and end goal city however for the longer paths with more options it is not as effective. 
# It does find the shortest path for Neamt especially because there are not too many options for what path the algorithm could look into taking. 
# It found the shortest path when going from Timisoara to Bucharest. 
# It did not find the least costy/shortest path when going from Oreda to Bucharest. It shows the smallest value for each pair of values but overall led it to still being a longer path. 

### BFS
# The BFS algorithm does work correctly and will return a path and cost for the graph, or will return an empty list and 0 otherwise.
# It does find the shortest path cost path from Timisoara to Bucharest, but performed better than DFS did. The path should have gone to Rimnicu Vilcea and Pitesti instead of Fagaras.
# It does find the shortest path cost path from Oradea to Bucharest, but performed much better then DFS. The issue was the same as above.
# It does find the shortest path from Neamt to Bucharest, and performed the same as DFS.

### A*
# The A* algorithm does work correctly and will return a path and cost for the graph
# It does find the shortest path for all 3 paths. outperforming the BFS and DFS
#

# -- Efficiency Discussion for each algorithm --
### DFS
# The alogorithm is more efficent for when there are fewer options or paths to take and gets less accuracte and efficent the more options you give it. 
# For the path from Oreda to Bucharest it chose the more costly path because the options presented to it were technically shorter but it had to go through more cities to get to Bucharest. 

### BFS
# It does not find the lowest cost path from Timisoara to Bucharest, but performed better than DFS did. The path should have gone to Rimnicu Vilcea and Pitesti instead of Fagaras.
# It does not find the lowest cost path from Oradea to Bucharest, but performed much better then DFS. The issue was the same as above.
# It does find the lowest cosst path from Neamt to Bucharest, and performed the same as DFS.
# The time complexity of BFS is O(V+E), where V is the vertices and E is the edges

### A*
# A* is an efficient and informed search algorithm that combines the advantages of both breadth-first search and the
# use of heuristics. It explores the most promising paths by considering a combination of the 
# cost to reach a node (g-value) and an estimate of the cost to reach the goal from 
# that node (heuristic, h-value). This approach minimizes the number of 
# nodes visited by prioritizing paths that are likely to lead to the goal. However, 
# A* efficiency can vary depending on the accuracy and admissibility of the heuristic,
# influencing the algorithm's ability to find the optimal solution in a timely manner.

### ChatGPT Disclaimer
# We utilized ChatGPT on this assignment as a way to check our work and the code we had written. It helped us to 
# better understand the algorithms themselves and how/why they are designed the way they are, especially A* search 
# which is newer to us. Along with this, we used it to adjust the comments we placed in our code to have a stronger, 
# more accurate description of the algorithm.
