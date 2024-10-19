# pyright: reportUnusedCallResult=false
import sys
from copy import deepcopy
from dataclasses import dataclass


def printerr(
    *args,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    **kwargs,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
) -> None:
    print(
        *args, file=sys.stderr, **kwargs  # pyright: ignore[reportUnknownArgumentType]
    )


@dataclass
class Service:
    id: int
    source: int
    destination: int
    num_edges: int
    wavelength_lower: int
    wavelength_upper: int
    edges: list[int]
    value: int
    dead = False

    def wavelength_size(self):
        return self.wavelength_upper - self.wavelength_lower + 1


@dataclass
class Edge:
    id: int
    source: int
    destination: int
    services: list[int]
    dead: bool = False


@dataclass
class Node:
    id: int
    converters: int


def input_c():
    return input().strip()


def read_environment():
    num_nodes, num_edges = input_c().split(" ")
    num_nodes, num_edges = int(num_nodes), int(num_edges)
    nodes = [Node(i + 1, converters=int(n)) for i, n in enumerate(input_c().split(" "))]
    if len(nodes) != num_nodes:
        raise ValueError("Number of nodes does not match number of channel conversions")
    edges: list[Edge] = [
        Edge(i + 1, int(v[0]), int(v[1]), [])
        for i, v in enumerate(input_c().split(" ") for _ in range(num_edges))
    ]
    num_services = int(input_c())
    services: list[Service] = []
    for i in range(num_services):
        (
            source,
            destination,
            num_edges,
            wavelength_lower,
            wavelength_upper,
            value,
        ) = [int(i) for i in input_c().split(" ")]

        service_edges = [int(n) for n in input_c().split(" ")]
        services.append(
            Service(
                id=i + 1,
                source=source,
                destination=destination,
                num_edges=num_edges,
                wavelength_lower=wavelength_lower,
                wavelength_upper=wavelength_upper,
                edges=service_edges,
                value=value,
            )
        )
        for edge in service_edges:
            edges[edge - 1].services.append(services[-1].id)
    return nodes, edges, services


INFINITY = float("inf")


class Graph:
    def __init__(self, nodes: list[Node], edges: list[Edge], services: list[Service]):
        self.nodes = nodes
        self.edges = edges
        self.services = services
        # source -> destination -> list[edge]
        self.edge_map: dict[int, dict[int, list[Edge]]] = {}

        for edge in edges:
            if edge.source not in self.edge_map:
                self.edge_map[edge.source] = {}
            if edge.destination not in self.edge_map:
                self.edge_map[edge.destination] = {}
            if self.edge_map[edge.source].get(edge.destination) is None:
                self.edge_map[edge.source][edge.destination] = []
            if self.edge_map[edge.destination].get(edge.source) is None:
                self.edge_map[edge.destination][edge.source] = []

            self.edge_map[edge.source][edge.destination].append(edge)
            self.edge_map[edge.destination][edge.source].append(edge)

        # Ensure that the edges are undirected
        e = edges[0]
        assert id(self.edge_map[e.source][e.destination][0]) == id(
            self.edge_map[e.destination][e.source][0]
        )

    def shortest_path(self, service: Service):
        unvisited_nodes = self.nodes.copy()
        distance_from_start: dict[int, float] = {
            node.id: (0 if node.id == service.source else INFINITY)
            for node in self.nodes
        }
        previous_node = {node.id: -1 for node in self.nodes}
        previous_edge = {node.id: -1 for node in self.nodes}

        while unvisited_nodes:
            current_node = min(
                unvisited_nodes, key=lambda node: distance_from_start[node.id]
            )
            unvisited_nodes.remove(current_node)

            if distance_from_start[current_node.id] == INFINITY:
                break

            for neighbor_node, neighbors in self.edge_map[current_node.id].items():
                for edge in neighbors:
                    if edge.dead:
                        continue
                    new_path = distance_from_start[current_node.id] + 1

                    constraint_violated = False
                    for other_service in edge.services:
                        # Check for overlapping wavelengths
                        if other_service == service.id:
                            continue
                        other_service = self.services[other_service - 1]
                        if (
                            other_service.wavelength_lower <= service.wavelength_upper
                            and other_service.wavelength_upper
                            >= service.wavelength_lower
                        ):
                            constraint_violated = True
                            break

                    if (
                        not constraint_violated
                        and new_path < distance_from_start[neighbor_node]
                    ):
                        distance_from_start[neighbor_node] = new_path
                        previous_node[neighbor_node] = current_node.id
                        previous_edge[neighbor_node] = edge.id
            if current_node.id == service.destination:
                break

        if previous_node[service.destination] == -1:
            return False

        # Reconstruct path
        path: list[int] = []
        current_node = service.destination
        while previous_edge[current_node] != -1:
            path.append(previous_edge[current_node])
            current_node = previous_node[current_node]
        path.reverse()
        return path


@dataclass
class Replan:
    service_idx: int
    edges: list[tuple[int, int, int]]


def get_replans(graph: Graph, broken_edge: int):
    # Find services affected by the broken edge
    affected_services: list[Service] = []
    for service in graph.services:
        if not service.dead and broken_edge in service.edges:
            service.dead = True
            affected_services.append(service)
    # Sort services by value (descending)
    affected_services.sort(key=lambda s: s.value, reverse=True)
    successful_replans: list[Replan] = []
    for service in affected_services:
        path = graph.shortest_path(service)
        if path is False:
            continue
        # Remove service from the edges
        for edge in service.edges:
            for s in graph.edges[edge - 1].services:
                if s == service.id:
                    graph.edges[edge - 1].services.remove(s)
                    break
        # Use up the channels
        for edge in path:
            edge = graph.edges[edge - 1]
            if service.id not in edge.services:
                edge.services.append(service.id)

        successful_replans.append(
            Replan(
                service_idx=service.id,
                edges=[
                    (e, service.wavelength_lower, service.wavelength_upper)
                    for e in path
                ],
            )
        )
        service.edges = [edge for edge in path]
        service.dead = False
    return successful_replans, affected_services


def get_recovery_rate(graph: Graph, edge: Edge) -> float:
    successful_replans, affected_services = get_replans(graph, edge.id)
    total_value = sum(s.value for s in affected_services)
    if total_value == 0:
        return 0
    return (
        sum(graph.services[s.service_idx - 1].value for s in successful_replans)
        / total_value
    )


def main():
    nodes_o, edges_o, services_o = read_environment()
    graph_o = Graph(nodes_o, edges_o, services_o)
    # Generate test scenarios by finding edges with highest recovery rate
    test_scenarios: list[list[int]] = []
    edges = deepcopy(edges_o)
    edges.sort(
        key=lambda e: sum(services_o[s - 1].value for s in e.services), reverse=True
    )
    # Take the top 30% of edges
    edges = edges[: int(len(edges) * 0.3)]
    edges.sort(
        key=lambda e: get_recovery_rate(deepcopy(graph_o), e),
        reverse=True,
    )
    # Take the top 30% of edges
    edges = edges[: int(len(edges) * 0.3)]
    edges.sort(
        key=lambda e: sum(services_o[s - 1].value for s in e.services), reverse=True
    )
    # Split into 29 scenarios
    num_edges_per_scenario = len(edges) // 29
    for i in range(29):
        test_scenarios.append(
            [
                e.id
                for e in edges[
                    i * num_edges_per_scenario : (i + 1) * num_edges_per_scenario
                ]
            ]
        )

    print(len(test_scenarios))  # Number of test scenarios
    for scenario in test_scenarios:
        print(len(scenario))
        print(" ".join(str(e) for e in scenario))

    num_scenarios = int(input_c())

    for _ in range(num_scenarios):
        graph = deepcopy(graph_o)
        while True:
            broken_edge = int(input_c())
            if broken_edge == -1:
                break
            graph.edges[broken_edge - 1].dead = True

            successful_replans, _ = get_replans(graph, broken_edge)

            print(len(successful_replans))
            for replan in successful_replans:
                print(replan.service_idx, len(replan.edges))
                print(" ".join(f"{e[0]} {e[1]} {e[2]}" for e in replan.edges))
    printerr("Done")


if __name__ == "__main__":
    main()
