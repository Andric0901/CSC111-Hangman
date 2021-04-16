"""The Hangman GameGraph"""

from __future__ import annotations

VALID_CHARACTERS = 'abcdefghijklmnopqrstuvwxyz'


class _VertexWeight:
    """Data class for a weighted vertex in a directed graph.

    Edges are outgoing.
    Weight of each edge is relative to total_weight.

    Representation invariant:
        - sum(self.edges.values()) == self.total_weight
    """
    total_weight: float
    edges: dict[str, float]

    def __init__(self) -> None:
        """Initialize a new vertex."""
        self.total_weight = 0
        self.edges = {}


class GameGraph:
    """The Hangman GameGraph.

    Each node represents a character in a word.
    This graph is directed.
    """

    # Private Instance Attribute
    #   - _vertices: collection of vertices contained in this graph
    _verts: dict[str, _VertexWeight]

    def __init__(self) -> None:
        """Initializes an empty GameGraph."""
        self._verts = {}

    def add_vertex(self, item: str) -> None:
        """Add a vertex with the given item to this graph."""
        if item not in self._verts:
            self._verts[item] = _VertexWeight()

    def __contains__(self, item: str) -> bool:
        """Check if item is in graph"""
        return item in self._verts

    def accumulate_edge(self, item1: str, item2: str, weight: float = 1) -> None:
        """Adds weight to the DIRECTED edge between item1 and item2.

        Preconditions:
            - item1 and item2 are in the graph
            - item1 != item2
        """
        v1 = self._verts[item1]
        v2 = self._verts[item2]

        v1.total_weight += weight

        if v2 in v1.edges:
            v1.edges[item2] += weight
        else:
            v1.edges[item2] = weight

    def get_weight(self, item1: str, item2: str) -> float:
        """Return the weight of the DIRECTED edge from item1 to item2

        Return 0 if item1 and item2 are not adjacent.

        Preconditions:
            - item1 and item2 are in the graph
        """
        v1 = self._verts[item1]
        return v1.edges.get(item2, 0)

    def get_vertex_weight(self, item: str) -> float:
        """Return total weight of item node"""
        return self._verts[item].total_weight

    def get_all_vertices(self) -> set:
        """Return all vertex items"""
        return set(self._verts.keys())

    def get_neighbours(self, item: str) -> set[str]:
        """Return the neighbours of item vertex"""
        return set(self._verts[item].edges.keys())

    def insert_character_sequence(self, characters: str) -> None:
        """Insert the given sequence of characters into this tree.

        >>> game = GameGraph()
        >>> game.insert_character_sequence('test')
        >>> game.insert_character_sequence('interesting')
        >>> game.get_weight('e', 's')
        2
        >>> game.get_weight('e', 'r')
        1
        """
        for c in characters:
            self.add_vertex(c)

        for i in range(len(characters) - 1):
            self.accumulate_edge(characters[i], characters[i + 1], 1)

