# pyright: reportImplicitRelativeImport=false
from dataclasses import dataclass
from random import randint

from main import find_available_wavelengths, merge_wavelenghts

N_EDGES = 1000
N_NODES = 100
N_SERVICES = 1000


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


@dataclass
class Service:
    id: int
    source: int
    destination: int
    edges: list[int]
    starting_wavelength: WavelengthRange
    value: int


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
                wl = find_available_wavelengths(occupied, wavelength_size)
                pwl = previous_wl.get(current.id)
                if pwl is not None:
                    wl = merge_wavelenghts(wl, pwl, wavelength_size)
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
    edges: dict[int, dict[int, Edge]] = {}
    # Ensure that every node has at least one edge
    for node in nodes:
        edges[node.id] = {}
        destination = randint(1, len(nodes))
        while destination == node.id:
            destination = randint(1, len(nodes))
        edges[node.id][destination] = Edge(0, node.id, destination, [])
        edges[destination] = {}
        edges[destination][node.id] = Edge(0, destination, node.id, [])
    for _ in range(N_EDGES - 2 * len(nodes)):
        source = randint(1, len(nodes))
        destination = randint(1, len(nodes))
        if source == destination:
            continue
        if destination not in edges[source]:
            edges[source][destination] = Edge(0, source, destination, [])
            edges[destination][source] = Edge(0, destination, source, [])
    edges_list: list[Edge] = []
    for edges_dict in edges.values():
        edges_list.extend(edges_dict.values())

    # Ensure that the edges are sorted by id and contiguous
    edges_list.sort(key=lambda edge: edge.id)
    for i, edge in enumerate(edges_list):
        edge.id = i + 1

    return edges_list


def new_graph():
    nodes = generate_nodes()
    edges = generate_edges(nodes)
    return Graph(nodes, edges, [])


def generate_services(graph: Graph):
    success = 1
    for _ in range(1, N_SERVICES + 1):
        src = randint(1, len(graph.nodes))
        dest = randint(1, len(graph.nodes))
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
    graph = new_graph()
    generate_services(graph)
