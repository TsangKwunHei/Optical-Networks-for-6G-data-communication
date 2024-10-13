# pyright: reportUnusedCallResult=false
import sys
from collections import deque
from copy import deepcopy
from dataclasses import dataclass


def printerr(
    *args,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    **kwargs,  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
):
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
        # Make undirected edges
        self.edge_map: dict[int, dict[int, Edge]] = {}

        for edge in edges:
            if edge.source not in self.edge_map:
                self.edge_map[edge.source] = {}
            if edge.destination not in self.edge_map:
                self.edge_map[edge.destination] = {}

            self.edge_map[edge.source][edge.destination] = edge
            self.edge_map[edge.destination][edge.source] = edge

        # Ensure that the edges are undirected
        e = edges[0]
        assert id(self.edge_map[e.source][e.destination]) == id(
            self.edge_map[e.destination][e.source]
        )

    def shortest_path(self, service: Service):
        unvisited_nodes = self.nodes.copy()
        distance_from_start: dict[int, float] = {
            node.id: (0 if node.id == service.source else INFINITY)
            for node in self.nodes
        }
        previous_node = {node.id: -1 for node in self.nodes}
        previous_node_wavelengths: dict[int, list[tuple[int, int]]] = {
            node.id: [] for node in self.nodes
        }

        while unvisited_nodes:
            current_node = min(
                unvisited_nodes, key=lambda node: distance_from_start[node.id]
            )
            unvisited_nodes.remove(current_node)

            if distance_from_start[current_node.id] == INFINITY:
                break

            for neighbor in self.edge_map[current_node.id]:
                if self.edge_map[current_node.id][neighbor].dead:
                    continue
                new_path = distance_from_start[current_node.id] + 1
                edge = self.edge_map[current_node.id][neighbor]

                available_wavelengths = self.find_available_wavelengths(
                    edge, service.wavelength_size()
                )

                if previous_node[current_node.id] != -1:

                    available_wavelengths = self.merge_wavelenghts(
                        available_wavelengths,
                        previous_node_wavelengths[current_node.id],
                        service.wavelength_size(),
                    )

                # Check for service wavelength constraints
                constraint_violated = len(available_wavelengths) == 0

                if not constraint_violated and new_path < distance_from_start[neighbor]:
                    distance_from_start[neighbor] = new_path
                    previous_node[neighbor] = current_node.id
                    previous_node_wavelengths[neighbor] = available_wavelengths
            if current_node.id == service.destination:
                break

        if previous_node[service.destination] == -1:
            return False
        selected_wavelength: tuple[int, int] = previous_node_wavelengths[
            service.destination
        ][0]
        for wavelength in previous_node_wavelengths[service.destination]:
            # Minimize size of range left over
            wavelength_size = wavelength[1] - wavelength[0] + 1
            selected_wavelength_size = (
                selected_wavelength[1] - selected_wavelength[0] + 1
            )
            if wavelength_size < selected_wavelength_size:
                selected_wavelength = wavelength
        wavelength_size = service.wavelength_size()
        service.wavelength_lower = selected_wavelength[0]
        service.wavelength_upper = selected_wavelength[0] + wavelength_size - 1

        # Reconstruct path
        path: deque[Edge] = deque()
        current_node = service.destination
        while previous_node[current_node] != -1:
            path.appendleft(self.edge_map[previous_node[current_node]][current_node])
            current_node = previous_node[current_node]
        return path

    def find_available_wavelengths(self, edge: Edge, min_size: int):
        occupied = sorted(
            [
                (
                    self.services[s - 1].wavelength_lower,
                    self.services[s - 1].wavelength_upper,
                )
                for s in edge.services
            ]
        )
        available_wavelengths: list[tuple[int, int]] = []
        if not occupied:
            available_wavelengths.append((1, 40))
            return available_wavelengths
        current = 1
        for lower, upper in occupied:
            if current < lower and lower - current >= min_size:
                available_wavelengths.append((current, lower - 1))
            current = max(current, upper + 1)
        if current <= 40 and 41 - current >= min_size:
            available_wavelengths.append((current, 40))
        return available_wavelengths

    def merge_wavelenghts(
        self, av_1: list[tuple[int, int]], av_2: list[tuple[int, int]], min_size: int
    ):
        # Shrink the list of available wavelengths such that only wavelengths available in both lists are kept
        result: list[tuple[int, int]] = []
        i, j = 0, 0
        while i < len(av_1) and j < len(av_2):
            # Find the overlapping range
            start = max(av_1[i][0], av_2[j][0])
            end = min(av_1[i][1], av_2[j][1])

            # If there's an overlap and it meets the minimum size requirement
            if start <= end and end - start + 1 >= min_size:
                result.append((start, end))

            # Move to the next range in the list with the smaller end point
            if av_1[i][1] < av_2[j][1]:
                i += 1
            else:
                j += 1

        return result


@dataclass
class Replan:
    service_idx: int
    edges: list[tuple[int, int, int]]


def main():
    nodes_o, edges_o, services_o = read_environment()
    print(1)  # Number of test scenarios
    print(2)
    print("1 6")

    num_scenarios = int(input_c())

    for _ in range(num_scenarios):
        services = deepcopy(services_o)
        graph = Graph(deepcopy(nodes_o), deepcopy(edges_o), services)
        while True:
            broken_edge = int(input_c())
            if broken_edge == -1:
                break
            graph.edges[broken_edge - 1].dead = True
            # Find services affected by the broken edge
            affected_services: list[Service] = []
            for service in services:
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
                    if service.id not in edge.services:
                        edge.services.append(service.id)

                successful_replans.append(
                    Replan(
                        service_idx=service.id,
                        edges=[
                            (e.id, service.wavelength_lower, service.wavelength_upper)
                            for e in path
                        ],
                    )
                )
                service.edges = [edge.id for edge in path]
                service.dead = False
            print(len(successful_replans))
            for replan in successful_replans:
                print(replan.service_idx, len(replan.edges))
                print(" ".join(f"{e[0]} {e[1]} {e[2]}" for e in replan.edges))


main()
