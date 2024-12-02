import networkx as nx
import random
import matplotlib.pyplot as plt
import heapq
from collections import deque
import time
import tracemalloc

def create_graph(node_count=20, edge_count=55, weight_range=(1, 16)):
    """
    Creates a directed graph with the exact specified number of edges, avoiding self-loops,
    duplicate edges, and ensuring no node is isolated.
    
    Parameters:
        node_count (int): Number of nodes in the graph.
        edge_count (int): Number of edges in the graph.
        weight_range (tuple): Range of weights for the edges (min_weight, max_weight).
    
    Returns:
        graph (nx.DiGraph): A directed graph with the specified parameters.
    """
    max_possible_edges = node_count * (node_count - 1)
    if edge_count > max_possible_edges:
        raise ValueError("Too many edges for the number of nodes in a directed graph.")
    if edge_count < node_count - 1:
        raise ValueError("Too few edges to ensure no node is isolated.")

    graph = nx.DiGraph()
    graph.add_nodes_from(range(1, node_count + 1))

    # Generate all possible directed edges avoiding self-loops
    all_possible_edges = [(u, v) for u in range(1, node_count + 1) for v in range(1, node_count + 1) if u != v]

    # Start with a random spanning tree to ensure connectivity
    spanning_tree_edges = []
    nodes = list(range(1, node_count + 1))
    random.shuffle(nodes)
    for i in range(node_count - 1):
        u = nodes[i]
        v = nodes[i + 1]
        spanning_tree_edges.append((u, v))

    # Assign random weights to the spanning tree edges
    weighted_spanning_tree_edges = [(u, v, random.randint(*weight_range)) for u, v in spanning_tree_edges]

    # Add spanning tree edges to the graph
    graph.add_weighted_edges_from(weighted_spanning_tree_edges)

    # Remove used edges from the pool of possible edges
    used_edges = set(spanning_tree_edges)
    available_edges = [(u, v) for u, v in all_possible_edges if (u, v) not in used_edges]

    # Randomly sample the remaining edges to reach the exact edge count
    remaining_edge_count = edge_count - len(spanning_tree_edges)
    additional_edges = random.sample(available_edges, remaining_edge_count)
    weighted_additional_edges = [(u, v, random.randint(*weight_range)) for u, v in additional_edges]

    # Add the additional edges to the graph
    graph.add_weighted_edges_from(weighted_additional_edges)

    return graph


def display_graph(graph):
    """
    Displays a directed graph with weights, ensuring arrowheads are visible and edges are not obscured by node circles.
    
    Parameters:
        graph (nx.DiGraph): The directed graph to be displayed.
    """
    pos = nx.spring_layout(graph)  # Position nodes for visualization
    plt.figure(figsize=(10, 8))

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=1200)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')

    # Handle edges
    base_arc_rad = 0.2  # Base curvature for edges
    arrow_shift = 0.1   # Additional offset for arrowhead visibility
    for (u, v) in graph.edges():
        # Check for bidirectional edges
        if graph.has_edge(v, u):  # Bidirectional edges
            if u < v:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=[(u, v)],
                    connectionstyle=f"arc3,rad={base_arc_rad}",
                    edge_color="gray",
                    arrowsize=20,
                    min_source_margin=15,
                    min_target_margin=15,
                )
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=[(v, u)],
                    connectionstyle=f"arc3,rad=-{base_arc_rad * 1.5}",
                    edge_color="gray",
                    arrowsize=20,
                    min_source_margin=15,
                    min_target_margin=15,
                )
        else:  # Single-direction edges
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[(u, v)],
                connectionstyle=f"arc3,rad=0",
                edge_color="gray",
                arrowsize=20,
                min_source_margin=15,
                min_target_margin=15,
            )

    # Handle edge labels
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_color="red",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white"),
    )

    plt.title("Directed Graph with Weights", fontsize=16)
    plt.show()


def dijkstra(graph, source, target):
    """
    Finds the shortest path and its cost from source to target using Dijkstra's algorithm.
    
    Parameters:
        graph (nx.DiGraph): The graph to search, with edge weights.
        source (int): The starting node.
        target (int): The target node.
    
    Returns:
        tuple: A tuple containing the shortest distance and the path as a list of nodes.
    """
    # Priority queue to store (distance, node)
    priority_queue = [(0, source)]
    distances = {node: float('inf') for node in graph.nodes}
    previous_nodes = {node: None for node in graph.nodes}
    distances[source] = 0

    while priority_queue:
        # Pop the node with the smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # If we reach the target node, build the path and return
        if current_node == target:
            path = []
            while current_node:
                path.append(current_node)
                current_node = previous_nodes[current_node]
            return current_distance, path[::-1]

        # Skip processing if a shorter path to the current node has already been found
        if current_distance > distances[current_node]:
            continue

        # Check all neighbors of the current node
        for neighbor, edge_data in graph[current_node].items():
            weight = edge_data['weight']
            distance = current_distance + weight

            # If a shorter path to the neighbor is found, update it
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # If the target node is unreachable, return infinity and an empty path
    return float('inf'), []

def bfs(graph, source, target):
    """
    Finds the shortest path from source to target using Breadth-First Search (BFS).

    Parameters:
        graph (nx.DiGraph): The directed graph with weighted edges.
        source (int): The starting node.
        target (int): The target node.

    Returns:
        tuple: A tuple containing the shortest path as a list of nodes and the total path cost,
               or (None, None) if no path exists.
    """
    if source not in graph.nodes or target not in graph.nodes:
        raise ValueError("Source or target node is not in the graph.")
    
    # Queue for BFS: stores (node, path_to_node, path_cost)
    queue = deque([(source, [source], 0)])  # Start with cost 0
    
    # Visited set to avoid re-processing nodes
    visited = set()

    while queue:
        current_node, path, path_cost = queue.popleft()

        # If the target node is found, return the path and the total cost
        if current_node == target:
            return path, path_cost

        # Mark the node as visited
        visited.add(current_node)

        # Add neighbors to the queue if they haven't been visited
        for neighbor in graph.successors(current_node):
            if neighbor not in visited:
                edge_weight = graph[current_node][neighbor]['weight']  # Get the weight of the edge
                queue.append((neighbor, path + [neighbor], path_cost + edge_weight))
                visited.add(neighbor)  # Mark as visited when adding to the queue to avoid re-processing

    # If no path is found
    return None, None


def traffic_based_heuristic(graph, current, target):
    """
    A heuristic function estimating the cost from the current node to the target.

    Parameters:
        graph (nx.DiGraph): The graph representing the road network.
        current (int): The current node.
        target (int): The target node.
    
    Returns:
        estimated_cost (float): Estimated cost from current to target.
    """
    # Use edge weights as a proxy for traffic conditions
    def average_outgoing_weight(node):
        edges = graph.out_edges(node, data=True)
        return sum(data['weight'] for _, _, data in edges) / len(edges) if edges else 0

    # Calculate average weights for current and target nodes
    current_traffic = average_outgoing_weight(current)
    target_traffic = average_outgoing_weight(target)

    # Estimate cost using traffic and approximate straight-line distance
    estimated_cost = current_traffic + target_traffic + abs(current - target)

    return estimated_cost


def astar(graph, source, target, heuristic):
    """
    Finds the shortest path from source to target using the A* algorithm.

    Parameters:
        graph (nx.DiGraph): The graph to search.
        source (int): The starting node.
        target (int): The goal node.
        heuristic (callable): A function that estimates the cost from a node to the target.
                             It should accept two arguments: the current node and the target node.
    
    Returns:
        tuple: The shortest path from source to target and the total path cost.
    """
    # Priority queue to hold the nodes to explore
    open_set = []
    heapq.heappush(open_set, (0, source))  # (priority, node)
    
    # Dictionary to track the cost of the shortest path to each node
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[source] = 0
    
    # Dictionary to store the estimated cost (g + h)
    f_score = {node: float('inf') for node in graph.nodes}
    f_score[source] = heuristic(graph, source, target)
    
    # Dictionary to reconstruct the path
    came_from = {}
    
    while open_set:
        # Get the node with the lowest f_score
        _, current = heapq.heappop(open_set)
        
        # If we reached the target, reconstruct and return the path
        if current == target:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(source)
            path.reverse()
            return path, g_score[target]
        
        # Explore neighbors
        for neighbor in graph.neighbors(current):
            edge_weight = graph[current][neighbor]['weight']
            tentative_g_score = g_score[current] + edge_weight
            
            if tentative_g_score < g_score[neighbor]:
                # This path to the neighbor is better than previously found
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(graph, neighbor, target)
                
                # Add the neighbor to the open set if it's not already there
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # If we exhaust the open set without finding the target
    return None, float('inf')


def measure_performance(algorithm, graph, source, target, heuristic=None):
    """
    Measures the runtime, memory usage, and performance of a given algorithm.

    Parameters:
        algorithm (callable): The algorithm function to evaluate.
        graph (any): The graph or data structure used by the algorithm.
        source (any): The starting node for the algorithm.
        target (any): The target node for the algorithm.
        heuristic (callable, optional): A heuristic function to pass to the algorithm.

    Returns:
        dict: A dictionary containing runtime, memory usage, result, and other performance details.
    """
    tracemalloc.start()  # Start tracing memory usage

    start_time = time.perf_counter()  # Start timer
    if heuristic:
        result = algorithm(graph, source, target, heuristic)  # Pass the heuristic
    else:
        result = algorithm(graph, source, target)  # No heuristic needed
    end_time = time.perf_counter()  # End timer

    current_memory, peak_memory = tracemalloc.get_traced_memory()  # Get memory usage
    tracemalloc.stop()  # Stop tracing memory usage

    performance = {
        "runtime": end_time - start_time,
        "current_memory": current_memory / 1024,  # Convert bytes to KB
        "peak_memory": peak_memory / 1024,  # Convert bytes to KB
        "result": result
    }
    return performance


# Uncomment the below to see results if you are running only this script. 


# # Example usage:
# node_count = 20
# edge_count = 55
# graph = create_graph(node_count, edge_count)
# graph = create_graph()
# display_graph(graph)
# source = 1
# target = 19

# # To visualize or inspect the graph:
# print("Graph edges with weights:")
# for edge in graph.edges(data=True):
#     print(edge)

# print()
# print()
# print("Results of Dijkstra's algorithm")
# print()
# shortest_distance, shortest_path = dijkstra(graph, source, target)
# print(f"Shortest distance from {source} to {target}: {shortest_distance}")
# print(f"Shortest path from {source} to {target}: {shortest_path}")
# print()
# performance = measure_performance(dijkstra, graph, source, target)
# print(f"Runtime: {performance['runtime']:.10f} seconds")
# print(f"Current Memory Usage: {performance['current_memory']:.2f} KB")
# print(f"Peak Memory Usage: {performance['peak_memory']:.2f} KB")

# print()
# print()
# print("Results of BFS algorithm")
# print()
# shortest_path_bfs, shortest_distance_bfs = bfs(graph, source, target)
# print(f"Shortest distance from {source} to {target}: {shortest_distance_bfs}")
# print(f"Shortest path from {source} to {target}: {shortest_path_bfs}")
# print()
# performance = measure_performance(bfs, graph, source, target)
# print(f"Runtime: {performance['runtime']:.10f} seconds")
# print(f"Current Memory Usage: {performance['current_memory']:.2f} KB")
# print(f"Peak Memory Usage: {performance['peak_memory']:.2f} KB")

# print()
# print()
# print("Results of A* algorithm")
# print()
# shortest_path_astar, shortest_distance_astar = astar(graph, source, target, traffic_based_heuristic)
# print(f"Shortest distance from {source} to {target}:", shortest_distance_astar)
# print(f"Shortest path from {source} to {target}:", shortest_path_astar)
# print()
# performance = measure_performance(astar, graph, source, target, traffic_based_heuristic)
# print(f"Runtime: {performance['runtime']:.10f} seconds")
# print(f"Current Memory Usage: {performance['current_memory']:.2f} KB")
# print(f"Peak Memory Usage: {performance['peak_memory']:.2f} KB")


# print()

