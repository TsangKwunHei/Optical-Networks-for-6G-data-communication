def calculate_edge_usage(services):
    # Collect edges used by services, weighted by service value
    edge_usage = {}
    for service_id, service in services.items():
        for edge_id in service['path_edges']:
            edge_usage.setdefault(edge_id, 0)
            edge_usage[edge_id] += service['w']
    return edge_usage

def generate_initial_scenarios(edge_usage, k, m):
    print("\nGenerating bottleneck edge failure scenarios...")
    # Sort edges by usage (descending order)
    sorted_edges = sorted(edge_usage.items(), key=lambda x: -x[1])
    print(f"Number of test scenarios to generate: {k}")
    e_failed_list = []
    # Generate test scenarios with top used edges as failures
    for i in range(k):
        if i < len(sorted_edges):
            e_failed = [sorted_edges[i][0]]
        else:
            e_failed = [1]  # Default to edge 1 if not enough edges
        e_count = len(e_failed)
        print(f"Test scenario {i+1}:")
        print(f"Number of failed edges: {e_count}")
        print(f"Failed edges: {e_failed}")
        e_failed_list.append(set(e_failed))
    return e_failed_list

def adjust_scenarios_to_reduce_similarity(e_failed_list, m):
    # Ensure that Jaccard similarity between test scenarios ≤ 0.5
    print("\nChecking Jaccard similarity between test scenarios...")
    k = len(e_failed_list)
    for i in range(k):
        for j in range(i + 1, k):
            intersection = e_failed_list[i] & e_failed_list[j]
            union = e_failed_list[i] | e_failed_list[j]
            similarity = len(intersection) / len(union)
            print(f"Similarity between scenario {i+1} and {j+1}: {similarity}")
            if similarity > 0.5:
                # Adjust the second scenario
                for edge_id in range(1, m + 1):
                    if edge_id not in e_failed_list[i]:
                        e_failed_list[j] = {edge_id}
                        print(f"Adjusting scenario {j+1} to reduce similarity")
                        break
    return e_failed_list

def determine_bottleneck(n, m, ci, graph, services, edges):
    # Step 1: Calculate edge usage
    edge_usage = calculate_edge_usage(services)
    # Step 2: Generate initial test scenarios
    k = 2
    e_failed_list = generate_initial_scenarios(edge_usage, k, m)
    # Step 3: Adjust scenarios to ensure Jaccard similarity ≤ 0.5
    e_failed_list = adjust_scenarios_to_reduce_similarity(e_failed_list, m)
    # Output test scenarios
    print("\nFinal test scenarios:")
    print(k)
    for e_failed in e_failed_list:
        print(len(e_failed))
        print(' '.join(map(str, e_failed)))
    return e_failed_list