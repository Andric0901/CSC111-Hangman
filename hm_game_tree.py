"""The Hangman GameTree."""

from __future__ import annotations
from typing import Optional

GAME_START_CHARACTER = '*'
VALID_CHARACTERS = 'abcdefghijklmnopqrstuvwxyz'


class GameTree:
    """The Hangman GameTree.

    Each node represents a character in a word. The very first node of the GameTree will be
    GAME_START_MOVE.

    Instance Attributes:
        - character: a string representing the character of each node
        - frequency: an int representing the frequency of how many times a node has been called.

    Representation Invariants:
        - self.character == GAME_START_CHARACTER or self.character in VALID_CHARACTERS
        - self.frequency >= 1
    """
    character: str
    frequency: int

    # Private Instance Attribute
    #   - _subtrees: the subtrees of this tree, which represent the game trees after a possible
    #                guess by the current player
    _subtrees: list[GameTree]

    def __init__(self, character: str = GAME_START_CHARACTER) -> None:
        """Initializes a new GameTree.

        >>> game = GameTree()
        >>> game.character == GAME_START_CHARACTER
        True
        >>> game.frequency
        1
        """
        self.character = character
        self.frequency = 1
        self._subtrees = []

    def get_subtrees(self) -> list[GameTree]:
        """Return the subtrees of this game tree."""
        return self._subtrees

    def find_subtree_by_character(self, character: str) -> Optional[GameTree]:
        """Return the subtree corresponding to the given move.

        Return None if no subtree corresponds to that move.
        """
        for subtree in self._subtrees:
            if subtree.character == character:
                return subtree

        return None

    def add_subtree(self, subtree: GameTree) -> None:
        """Add a subtree to this game tree."""
        self._subtrees.append(subtree)

    def __str__(self) -> str:
        """Return a string representation of this tree.
        """
        return self._str_indented(0)

    def _str_indented(self, depth: int) -> str:
        """Return an indented string representation of this tree.

        The indentation level is specified by the <depth> parameter.
        """
        move_desc = f'{self.character}\n'
        s = '  ' * depth + move_desc
        if self._subtrees == []:
            return s
        else:
            for subtree in self._subtrees:
                s += subtree._str_indented(depth + 1)
            return s

    def insert_character_sequence(self, characters: list[str]) -> None:
        """Insert the given sequence of characters into this tree.

        The inserted characters form a chain of descendants, where:
            - characters[0] is a child of this tree's root
            - characters[1] is a child of characters[0]
            - characters[2] is a child of characters[1]
            - etc.
        """
        if characters == []:
            return
        elif self.find_subtree_by_character(characters[0]) is None:
            gt_free = GameTree(characters[0])
            gt_free.frequency += 1
            gt_free.insert_character_sequence(get_rest(characters))
            self.add_subtree(gt_free)
        else:
            existing_subtree = self.find_subtree_by_character(characters[0])
            existing_subtree.frequency += 1
            existing_subtree.insert_character_sequence(get_rest(characters))


# Leave this at the end of the file OUTSIDE of GameTree class
def get_rest(moves: list[str]) -> list[str]:
    """Return the list of 'rest', i.e., without the first element."""
    alt_moves = moves.copy()
    alt_moves.reverse()
    alt_moves.pop()
    alt_moves.reverse()
    return alt_moves


if __name__ == '__main__':
    import doctest
    doctest.testmod()
