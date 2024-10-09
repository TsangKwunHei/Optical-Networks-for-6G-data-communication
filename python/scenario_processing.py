# scenario_processing.py

def process_scenarios(n, m, ci, graph, services, e_failed_list):
    from service_replanning import replan_service

    print("\nProcessing test scenarios...")
    # Simulated interaction inputs
    interaction_inputs = [
        # First test scenario
        "1",  # Number of scenarios
        "1",  # Edge failed (edge id)
        "-1",  # End of scenario
    ]

    input_iter = iter(interaction_inputs)
    input = lambda: next(input_iter)

    k_input = int(input())
    print(f"Number of test scenarios to process: {k_input}")

    for scenario_idx in range(k_input):
        print(f"\nProcessing scenario {scenario_idx+1}...")
        # Reset the environment for each scenario
        # Copy ci and services
        ci_current = ci[:]
        services_current = {}
        for service_id, service in services.items():
            services_current[service_id] = {
                **service,
                'old_resources': {
                    'edges': set(service['old_resources']['edges']),
                    'wavelengths': set(service['old_resources']['wavelengths']),
                    'channel_conversions': set(),
                }
            }

        # Edge status: True if edge is up, False if down
        edge_status = [True] * (m + 1)  # Edge indices start from 1

        while True:
            e_failed_line = input()
            if e_failed_line.strip() == "":
                continue
            e_failed = int(e_failed_line)
            if e_failed == -1:
                print("End of scenario input")
                break
            print(f"Edge failed: {e_failed}")
            edge_status[e_failed] = False

            # Identify affected services
            affected_services = []
            for service_id, service in services_current.items():
                if e_failed in service['path_edges']:
                    affected_services.append(service_id)
            print(f"Affected services: {affected_services}")

            # Replan services
            replanned_services = []
            for service_id in affected_services:
                service = services_current[service_id]
                success, new_path, new_wavelengths, used_conversions = replan_service(
                    service_id, service, graph, edge_status, ci_current, services_current, n, m
                )
                if success:
                    print(f"Service {service_id} successfully replanned")
                    # Update service with new plan
                    service['path_edges'] = new_path
                    service['l'] = len(new_path)
                    service['f_start'] = min(new_wavelengths)
                    service['f_end'] = max(new_wavelengths)
                    service['old_resources']['edges'] = set(new_path)
                    service['old_resources']['wavelengths'] = new_wavelengths
                    service['old_resources']['channel_conversions'] = used_conversions
                    # Deduct used channel conversions
                    for node in used_conversions:
                        ci_current[node] -= 1
                    replanned_services.append(service_id)
                else:
                    print(f"Service {service_id} could not be replanned and dies")
                    # Service cannot be replanned (dies)
                    pass

            # Output number of services successfully replanned
            print(f"Number of services replanned: {len(replanned_services)}")
            # Output replanned services
            for service_id in replanned_services:
                service = services_current[service_id]
                # First line: service index and number of edges
                print(f"Service {service_id} new path length: {service['l']}")
                # Second line: edge indices and wavelength interval
                edge_str = ' '.join(map(str, service['path_edges']))
                print(f"Service {service_id} path edges: {edge_str}, wavelengths {service['f_start']}-{service['f_end']}")
        # End of scenario