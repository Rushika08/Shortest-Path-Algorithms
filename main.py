from student_submission import (
    create_graph,
    display_graph,
    dijkstra,
    bfs,
    astar,
    measure_performance,
    traffic_based_heuristic
)

# Example usage:
node_count = 20
edge_count = 55
graph = create_graph(node_count, edge_count)

# Visualize the graph
display_graph(graph)

source = 1
target = 19

# Print graph edges with weights
print("Graph edges with weights:")
for edge in graph.edges(data=True):
    print(edge)

print("-----------------------------------------------")
print()
print("Results of Dijkstra's algorithm")
print()
shortest_distance, shortest_path = dijkstra(graph, source, target)
print(f"Shortest distance from {source} to {target}: {shortest_distance}")
print(f"Shortest path from {source} to {target}: {shortest_path}")
performance = measure_performance(dijkstra, graph, source, target)
print(f"Runtime: {performance['runtime']:.10f} seconds")
print(f"Current Memory Usage: {performance['current_memory']:.2f} KB")
print(f"Peak Memory Usage: {performance['peak_memory']:.2f} KB")

print("-----------------------------------------------")
print()
print("Results of BFS algorithm")
print()
shortest_path_bfs, shortest_distance_bfs = bfs(graph, source, target)
print(f"Shortest distance from {source} to {target}: {shortest_distance_bfs}")
print(f"Shortest path from {source} to {target}: {shortest_path_bfs}")
performance = measure_performance(bfs, graph, source, target)
print(f"Runtime: {performance['runtime']:.10f} seconds")
print(f"Current Memory Usage: {performance['current_memory']:.2f} KB")
print(f"Peak Memory Usage: {performance['peak_memory']:.2f} KB")

print("-----------------------------------------------")
print()
print("Results of A* algorithm")
print()
shortest_path_astar, shortest_distance_astar = astar(graph, source, target, traffic_based_heuristic)
print(f"Shortest distance from {source} to {target}: {shortest_distance_astar}")
print(f"Shortest path from {source} to {target}: {shortest_path_astar}")
performance = measure_performance(astar, graph, source, target, traffic_based_heuristic)
print(f"Runtime: {performance['runtime']:.10f} seconds")
print(f"Current Memory Usage: {performance['current_memory']:.2f} KB")
print(f"Peak Memory Usage: {performance['peak_memory']:.2f} KB")
print()
