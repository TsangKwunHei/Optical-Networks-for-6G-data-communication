from dataclasses import dataclass


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


@dataclass
class Edge:
    id: int
    source: int
    destination: int
    channel_used: list[tuple[int, int]]
    dead: bool = False


@dataclass
class Node:
    id: int
    converters: int
