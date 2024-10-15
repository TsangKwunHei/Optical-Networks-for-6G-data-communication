# pyright: reportImplicitRelativeImport=false
from copy import deepcopy
from dataclasses import dataclass
from random import randint

from main import find_available_wavelengths, merge_wavelenghts

N_EDGES = 1000
N_NODES = 100
N_SERVICES = 700


@dataclass
class WavelengthRange:
    """
    Wavelength 0-40. Inclusive.
    """

    min: int
    max: int


@dataclass
class Converter:
    id: int
    service_id: int | None
    old: WavelengthRange | None
    new: WavelengthRange | None


@dataclass
class Node:
    id: int  # starts from 1
    converters: list[Converter]


@dataclass
class Edge:
    id: int
    source: int
    destination: int
    services: list[int]
    dead: bool = False


@dataclass
class Service:
    id: int
    source: int
    destination: int
    edges: list[int]
    starting_wavelength: WavelengthRange
    value: int
    dead: bool = False


@dataclass
class Graph:
    nodes: list[Node]
    edges: list[Edge]
    services: list[Service]

    def pathfind(self, src: int, dest: int, wavelength_size: int):
        """
        Returns a list of edges that form a path from src to dest.
        """
        unvisited_nodes = self.nodes.copy()
        distances = {node.id: float("inf") for node in self.nodes}
        distances[src] = 0
        previous = {node.id: -1 for node in self.nodes}
        previous_wl: dict[int, list[tuple[int, int]]] = {}
        edges: dict[int, dict[int, Edge]] = {}
        for edge in self.edges:
            if edge.source not in edges:
                edges[edge.source] = {}
            edges[edge.source][edge.destination] = edge
            if edge.destination not in edges:
                edges[edge.destination] = {}
            edges[edge.destination][edge.source] = edge
        while unvisited_nodes:
            current = min(unvisited_nodes, key=lambda node: distances[node.id])
            unvisited_nodes.remove(current)
            if current.id == dest:
                break

            for neighbor in edges[current.id]:
                alt = distances[current.id] + 1
                # Ensure that the edge still has available wavelengths
                edge = edges[current.id][neighbor]
                occupied = sorted(
                    [
                        (
                            self.services[s - 1].starting_wavelength.min,
                            self.services[s - 1].starting_wavelength.max,
                        )
                        for s in edge.services
                    ]
                )
                wl = find_available_wavelengths(occupied, wavelength_size + 1)
                pwl = previous_wl.get(current.id)
                if pwl is not None:
                    wl = merge_wavelenghts(wl, pwl, wavelength_size + 1)
                if not wl:
                    continue

                if alt < distances[neighbor]:
                    distances[neighbor] = alt
                    previous[neighbor] = current.id
                    previous_wl[neighbor] = wl
        # Ensure a path was found
        if previous[dest] == -1:
            return False
        path: list[int] = []
        current = dest
        while previous[current] != -1:
            path.append(edges[previous[current]][current].id)
            current = previous[current]
        path.reverse()
        # Take the merged available wavelengths
        available_wl = previous_wl.get(dest)
        if available_wl is None:
            raise ValueError("No available wavelengths")
        # Sort available wavelengths by size (smallest first)
        available_wl.sort(key=lambda wl: wl[1] - wl[0])
        return path, available_wl[0][0]

    def validate(self):
        # Ensure that every service is valid
        for service in self.services:
            if service.source < 1 or service.source > len(self.nodes):
                raise ValueError("Invalid source node")
            if service.destination < 1 or service.destination > len(self.nodes):
                raise ValueError("Invalid destination node")
            if (
                service.starting_wavelength.min <= 0
                or service.starting_wavelength.max > 40
            ):
                raise ValueError("Invalid wavelength range")
            if service.value < 0:
                raise ValueError("Invalid value")
            for edge_id in service.edges:
                # Ensure that wavelengths don't overlap with other services on the edge
                edge = self.edges[edge_id - 1]
                occupied = [
                    (
                        self.services[s - 1].starting_wavelength.min,
                        self.services[s - 1].starting_wavelength.max,
                    )
                    for s in edge.services
                    if s != service.id
                ]

                for wl in occupied:
                    if (
                        wl[0] <= service.starting_wavelength.min <= wl[1]
                        or wl[0] <= service.starting_wavelength.max <= wl[1]
                    ):
                        print(wl, service)
                        raise ValueError("Wavelength overlap")
                # Ensure that the edge has the service
                if service.id not in edge.services:
                    raise ValueError("Service not in edge")


def generate_nodes():
    nodes: list[Node] = []
    for i in range(1, N_NODES + 1):
        N_CONVERTERS = randint(0, 20)
        nodes.append(
            Node(
                id=i,
                converters=[
                    Converter(j + 1, None, None, None) for j in range(N_CONVERTERS)
                ],
            )
        )
    return nodes


def generate_edges(nodes: list[Node]) -> list[Edge]:
    edges: dict[int, dict[int, list[Edge]]] = {}
    # Ensure that every node has at least one edge
    for node in nodes:
        edges[node.id] = {}
        destination = randint(1, len(nodes))
        while destination == node.id:
            destination = randint(1, len(nodes))
        edges[node.id][destination] = [Edge(0, node.id, destination, [])]
        edges[destination] = {}
    for _ in range(N_EDGES - 2 * len(nodes)):
        source = randint(1, len(nodes))
        destination = randint(1, len(nodes))
        if source == destination:
            continue
        if destination not in edges[source]:
            edges[source][destination] = []
        edges[source][destination].append(Edge(0, source, destination, []))

    edges_list: list[Edge] = []
    for edges_dict in edges.values():
        edges_list.extend(
            [edge for edge_list in edges_dict.values() for edge in edge_list]
        )

    # Ensure that the edges are sorted by id and contiguous
    edges_list.sort(key=lambda edge: edge.id)
    for i, edge in enumerate(edges_list):
        edge.id = i + 1

    return edges_list


def new_graph():
    nodes = generate_nodes()
    edges = generate_edges(nodes)
    graph = Graph(nodes, edges, [])
    generate_services(graph)
    return graph


def generate_services(graph: Graph):
    success = 1
    for _ in range(1, N_SERVICES + 1):
        src = randint(1, len(graph.nodes))
        dest = randint(1, len(graph.nodes))
        if src == dest:
            continue
        wavelength_size = randint(1, 30)
        value = randint(0, 100000)
        edges = graph.pathfind(src, dest, wavelength_size)
        if not edges:
            continue
        edges, lower_wl = edges
        service = Service(
            success,
            src,
            dest,
            edges,
            WavelengthRange(lower_wl, lower_wl + wavelength_size),
            value,
        )
        graph.services.append(service)
        # Assign the service to the edges
        for edge_id in edges:
            graph.edges[edge_id - 1].services.append(success)
        success += 1


if __name__ == "__main__":
    ### ENVIRONMENT
    graph = new_graph()
    graph.validate()
    print(len(graph.nodes), len(graph.edges))
    print(" ".join(str(len(node.converters)) for node in graph.nodes))
    for edge in graph.edges:
        print(edge.source, edge.destination)
    print(len(graph.services))
    for service in graph.services:
        print(
            service.source,
            service.destination,
            len(service.edges),
            service.starting_wavelength.min,
            service.starting_wavelength.max,
            service.value,
        )
        print(" ".join(str(edge) for edge in service.edges))
    n_edges = len(graph.edges)
    n_nodes = len(graph.nodes)
    n_services = len(graph.services)
    ### Read user privded scenarios
    n_scenarios = int(input())
    test_scenarios: list[set[int]] = []
    for _ in range(n_scenarios):
        n_edge_failures = int(input())
        test_scenarios.append(set(map(int, input().split())))
        assert len(test_scenarios[-1]) == n_edge_failures
    n_generated_scenarios = randint(1, 100)
    for _ in range(n_generated_scenarios):
        n_edge_failures = randint(1, 60)
        test_scenarios.append(
            set([randint(1, n_edges) for _ in range(n_edge_failures)])
        )
    print(len(test_scenarios))
    original_graph = graph
    for scenario in test_scenarios:
        graph = deepcopy(original_graph)
        for edge in scenario:
            print(edge)
            graph.edges[edge - 1].dead = True
            affected_services = [
                service.id
                for service in graph.services
                if edge in service.edges and not service.dead
            ]
            for affected_service in affected_services:
                graph.services[affected_service - 1].dead = True
            n_successful_replans = int(input())
            if n_successful_replans > len(affected_services):
                raise ValueError("More services replanned than affected")
            for _ in range(n_successful_replans):
                service_idx, num_edges = map(int, input().split())
                replans = list(map(int, input().split()))
                assert len(replans) == num_edges * 3  # edge, min, max
                service = graph.services[service_idx - 1]
                service.dead = False
                for edge in service.edges:
                    graph.edges[edge - 1].services.remove(service.id)
                service.edges = replans[::3]
                for edge in service.edges:
                    graph.edges[edge - 1].services.append(service.id)
                service.starting_wavelength = WavelengthRange(replans[1], replans[2])
        graph.validate()

        print(-1)
