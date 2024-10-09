# service_replanning.py

def replan_service(service_id, service, graph, edge_status, ci_current, services_current, n, m):
    import heapq

    print(f"Replanning service {service_id} from {service['s']} to {service['t']}")
    s = service['s']
    t = service['t']
    bandwidth = service['bandwidth']
    old_edges = service['old_resources']['edges']
    old_wavelengths = service['old_resources']['wavelengths']
    w = service['w']

    # Prepare data structures
    dist = [float('inf')] * (n + 1)
    prev = [None] * (n + 1)
    dist[s] = 0
    heap = [(0, s)]

    while heap:
        d, u = heapq.heappop(heap)
        if dist[u] < d:
            continue
        if u == t:
            print(f"Reached destination {t}")
            break
        for v, edge_id in graph[u]:
            if not edge_status[edge_id]:
                continue  # Edge is down
            # Edge cost can be adjusted to prefer certain paths
            cost = 1  # Uniform cost
            if edge_id in old_edges:
                cost -= 0.1  # Prefer reusing old edges
            if dist[v] > dist[u] + cost:
                dist[v] = dist[u] + cost
                prev[v] = (u, edge_id)
                heapq.heappush(heap, (dist[v], v))

    if dist[t] == float('inf'):
        print(f"No path found for service {service_id}")
        return False, None, None, None  # No path found

    # Reconstruct path
    path_edges = []
    node = t
    while node != s:
        u, edge_id = prev[node]
        path_edges.append(edge_id)
        node = u
    path_edges.reverse()
    print(f"Found new path for service {service_id}: edges {path_edges}")

    # Assign wavelengths
    max_wavelength = 100  # Max wavelength index
    wavelengths_on_edge = {}
    for edge_id in path_edges:
        wavelengths_on_edge[edge_id] = set()

    # Collect wavelengths used by other services (on old and new paths)
    for other_service_id, other_service in services_current.items():
        if other_service_id == service_id:
            continue
        # Wavelengths used on old paths
        for edge_id in other_service['old_resources']['edges']:
            if edge_id in wavelengths_on_edge:
                wavelengths_on_edge[edge_id].update(other_service['old_resources']['wavelengths'])
        # Wavelengths used on new paths (if replanned)
        if 'path_edges' in other_service:
            for edge_id in other_service['path_edges']:
                if edge_id in wavelengths_on_edge:
                    wavelengths_on_edge[edge_id].update(range(other_service['f_start'], other_service['f_end'] + 1))

    # Wavelengths available on each edge
    available_wavelengths = {}
    for edge_id in path_edges:
        occupied = wavelengths_on_edge[edge_id]
        # Remove service's own old wavelengths (can be reused)
        if edge_id in old_edges:
            occupied -= old_wavelengths
        available = set(range(1, max_wavelength + 1)) - occupied
        available_wavelengths[edge_id] = available

    # Try to find common wavelength intervals
    common_wavelengths = set.intersection(*available_wavelengths.values())
    # Include service's own old wavelengths (can be reused)
    for edge_id in path_edges:
        if edge_id in old_edges:
            common_wavelengths.update(old_wavelengths)

    possible_intervals = []
    common_list = sorted(common_wavelengths)
    for i in range(len(common_list)):
        start_wl = common_list[i]
        end_wl = start_wl + bandwidth - 1
        if end_wl > max_wavelength:
            break
        if all(wl in common_wavelengths for wl in range(start_wl, end_wl + 1)):
            possible_intervals.append((start_wl, end_wl))

    if possible_intervals:
        # Assign first possible interval
        f_start, f_end = possible_intervals[0]
        new_wavelengths = set(range(f_start, f_end + 1))
        used_conversions = set()
        print(f"Assigned wavelengths {f_start}-{f_end} to service {service_id}")
        return True, path_edges, new_wavelengths, used_conversions

    # If no common interval, try with channel conversions
    print(f"Trying to assign wavelengths with channel conversions for service {service_id}")
    # Initialize wavelengths per edge
    wavelengths_per_edge = []
    for edge_id in path_edges:
        available = available_wavelengths[edge_id]
        # Include own old wavelengths
        if edge_id in old_edges:
            available.update(old_wavelengths)
        available_list = sorted(available)
        intervals = []
        for i in range(len(available_list)):
            start_wl = available_list[i]
            end_wl = start_wl + bandwidth - 1
            if end_wl > max_wavelength:
                break
            if all(wl in available for wl in range(start_wl, end_wl + 1)):
                intervals.append((start_wl, end_wl))
        if not intervals:
            print(f"Cannot assign wavelengths on edge {edge_id} for service {service_id}")
            return False, None, None, None  # Cannot assign wavelengths
        wavelengths_per_edge.append(intervals)

    # Try to assign wavelengths, using channel conversions where needed
    assigned_intervals = []
    used_conversions = set()
    nodes_in_path = [service['s']]
    node = service['s']
    for edge_id in path_edges:
        for v, eid in graph[node]:
            if eid == edge_id:
                node = v
                break
        nodes_in_path.append(node)

    # Start with first edge's intervals
    assigned_intervals.append(wavelengths_per_edge[0][0])
    for i in range(1, len(path_edges)):
        prev_interval = assigned_intervals[-1]
        current_options = wavelengths_per_edge[i]
        # Check if same interval is available
        match = False
        for interval in current_options:
            if interval == prev_interval:
                assigned_intervals.append(interval)
                match = True
                break
        if not match:
            # Need channel conversion at node i
            node = nodes_in_path[i]
            if ci_current[node] > 0:
                assigned_intervals.append(current_options[0])
                used_conversions.add(node)
                print(f"Using channel conversion at node {node} for service {service_id}")
            else:
                print(f"No channel conversion available at node {node} for service {service_id}")
                return False, None, None, None  # No channel conversion available

    # Deduct used channel conversions
    for node in used_conversions:
        ci_current[node] -= 1

    # Collect wavelengths used
    new_wavelengths = set()
    for interval in assigned_intervals:
        new_wavelengths.update(range(interval[0], interval[1] + 1))

    print(f"Assigned wavelengths with conversions for service {service_id}")
    return True, path_edges, new_wavelengths, used_conversions