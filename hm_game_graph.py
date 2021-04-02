"""The Hangman GameGraph"""

from __future__ import annotations
from typing import Optional

VALID_CHARACTERS = 'abcdefghijklmnopqrstuvwxyz'


class _WeightedVertex:
    """A weighted vertex representing a character

    Neighbours are letters that appear next to this character
    Weight of each neighbour is relative to total
    
    Representation invariant:
        - sum(neighbours.values()) == total_weight
    """
    item: str
    total_weight: float
    neighbours: dict[_WeightedVertex, float]

    def __init__(self, item: str) -> None:
        """Initialize a new vertex with the given item and kind.

        This vertex is initialized with no neighbours."""
        self.item = item
        self.total_weight = 0
        self.neighbours = {}


class GameGraph:
    """The Hangman GameGraph.

    Each node represents a character in a word.
    This graph is directed.
    """

    # Private Instance Attribute
    #   - _vertices: collection of vertices contained in this graph
    _vertices: dict[str, _WeightedVertex]

    def __init__(self) -> None:
        """Initializes an empty GameGraph.

        >>> game = GameGraph()
        """
        self._vertices = {}
    
    def add_vertex(self, item: str) -> None:
        """Add a vertex with the given item to this graph.

        The new vertex is not adjacent to any other vertices.
        Do nothing if the given item is already in this graph.
        """
        if item not in self._vertices:
            self._vertices[item] = _WeightedVertex(item)

    def __contains__(self, item: str) -> bool:
        """Check if item is in graph"""
        return item in self._vertices

    def accumulate_edge(self, item1: str, item2: str, weight: float = 1) -> None:
        """Adds value to the DIRECTED edge between the two vertices in this graph.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        Preconditions:
            - item1 != item2
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            v2 = self._vertices[item2]

            v1.total_weight += weight

            if v2 in v1.neighbours:
                v1.neighbours[v2] += weight
            else:
                v1.neighbours[v2] = weight
        else:
            raise ValueError

    def get_weight(self, item1: str, item2: str) -> float:
        """Return the weight of the DIRECTED edge from item1 to item2

        Return 0 if item1 and item2 are not adjacent.

        Preconditions:
            - item1 and item2 are vertices in this graph
        """
        v1 = self._vertices[item1]
        v2 = self._vertices[item2]
        return v1.neighbours.get(v2, 0)

    def get_vertex_weight(self, item: str) -> float:
        """Return total weight of item node"""
        return self._vertices[item].total_weight

    def get_all_vertices(self) -> set:
        """Return all vertex items"""
        return set(self._vertices.keys())

    def get_neighbours(self, item: str) -> dict:
        """Return the neighbours of item node"""
        return {v.item for v in self._vertices[item].neighbours.keys()}

    def insert_character_sequence(self, characters: list[str]) -> None:
        """Insert the given sequence of characters into this tree.

        >>> game = GameGraph()
        >>> game.insert_character_sequence(list('test'))
        >>> game.insert_character_sequence(list('interesting'))
        >>> game.get_weight('e', 's')
        2
        >>> game.get_weight('e', 'r')
        1
        """
        for c in characters:
            self.add_vertex(c)
        
        for i in range(len(characters) - 1):
            self.accumulate_edge(characters[i], characters[i+1], 1)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
