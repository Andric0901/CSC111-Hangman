"""All Hangman AI Players."""

import csv
import random
from typing import Optional

import hm_game_tree
import hm_game_graph
import hangman

VALID_CHARACTERS = 'abcdefghijklmnopqrstuvwxyz'


def load_word_bank(games_file: str) -> hm_game_graph.GameGraph:
    """Create a word bank (i.e., a game graph) based on games_file."""
    empty_game_graph = hm_game_graph.GameGraph()
    with open(games_file) as file:
        for row in file:
            empty_game_graph.insert_character_sequence(row.strip('\n'))

    return empty_game_graph


class RandomPlayer(hangman.Player):
    """A Hangman AI whose strategy is always picking a random character."""

    def make_guess(self, game: hangman.Hangman, previous_character: Optional[str]) -> str:
        """Make a guess given the current game.

        previous_character is the player's most recently guessed character, or None if no guesses
        have been made.
        """
        chosen_character = random.choice(VALID_CHARACTERS)
        while chosen_character in self._visited_characters:
            chosen_character = random.choice(VALID_CHARACTERS)
        self._visited_characters.add(chosen_character)
        return chosen_character


class GraphNextPlayer(hangman.Player):
    """A Hangman player that plays based on a given GameGraph.

    This player uses a game graph to make guesses as the game is played.
    On its turn:

        1. If there are no known characters then it guesses the most frequent character
        2. If there are known characters then it guesses the next characters
        3. If it runs out of options then it guesses randomly
    """
    # Private Instance Attributes:
    #   - _game_graph:
    #       The GameGraph that this player uses to make its moves. If None, then this
    #       player just makes random moves.
    _graph: Optional[hm_game_graph.GameGraph]

    def __init__(self, graph: hm_game_graph.GameGraph) -> None:
        """Initialize this player.

        Preconditions:
            - graph represents a game graph
        """
        self._graph = graph
        self._visited_characters = set()

    def make_guess(self, game: hangman.Hangman, previous_move: Optional[str]) -> str:
        """Make a guess given the current game.

        previous_character is the player's most recently guessed character, or None if no guesses
        have been made.
        """

        status = game.get_guess_status()
        if status == '?' * len(status):
            # Beginning, choose most common character
            chars = {(w, self._graph.get_vertex_weight(w))
                     for w in self._graph.get_all_vertices()
                     if w not in self._visited_characters}
            choice = max(chars, key=lambda p: p[1])[0]
            self._visited_characters.add(choice)
            print('Beginning ', end='  ')
            return choice

        # Guess adjacent character (first one seen)
        for i in range(len(status) - 1):
            s = status[i]
            n = status[i + 1]
            if (s in VALID_CHARACTERS) and (n == '?') and (s in self._graph):
                chars = {(w, self._graph.get_weight(s, w))
                          for w in self._graph.get_neighbours(s)
                          if w not in self._visited_characters}
                if len(chars) > 0:
                    choice = max(chars, key=lambda p: p[1])[0]
                    self._visited_characters.add(choice)
                    print('Adjacent', s, end='  ')
                    return choice

        # Last resort, random guess
        char = random.choice(VALID_CHARACTERS)
        while char in self._visited_characters:
            char = random.choice(VALID_CHARACTERS)
        self._visited_characters.add(char)
        print('Random    ', end='  ')
        return char


class RandomTreePlayer(hangman.Player):
    """A Hangman player that plays randomly based on a given GameTree.

    This player uses a game tree to make guesses, descending into the tree as the game is played.
    On its turn:

        1. First it updates its game tree to its subtree corresponding to the guess made by
           the AI. If no subtree is found, its game tree is set to None.
        2. Then, if its game tree is not None, it picks its next character randomly from among
           the subtrees of its game tree, and then reassigns its game tree to that subtree.
           But if its game tree is None or has no subtrees, the player picks its next
           character randomly, and then sets its game tree to None.
    """
    # Private Instance Attributes:
    #   - _game_tree:
    #       The GameTree that this player uses to make its moves. If None, then this
    #       player just makes random moves.
    _game_tree: Optional[hm_game_tree.GameTree]

    def __init__(self, game_tree: hm_game_tree.GameTree) -> None:
        """Initialize this player.

        Preconditions:
            - game_tree represents a game tree at the initial state (root is '*')
        """
        self._game_tree = game_tree

    def make_guess(self, game: hangman.Hangman, previous_move: Optional[str]) -> str:
        """Make a guess given the current game.

        previous_character is the player's most recently guessed character, or None if no guesses
        have been made.
        """
        if previous_move is None:
            pass
        elif self._game_tree is not None and \
                self._game_tree.find_subtree_by_character(previous_move) is not None:
            self._game_tree = self._game_tree.find_subtree_by_character(previous_move)
        else:
            self._game_tree = None

        if self._game_tree is not None:
            subtrees = self._game_tree.get_subtrees()
            # TODO: loop over the subtrees, check each subtree so that
            #       the characters are not in self._visited_characters
            chosen_subtree = random.choice(subtrees)
            self._game_tree = chosen_subtree
            return chosen_subtree.character
        else:
            chosen_character = random.choice(VALID_CHARACTERS)
            while chosen_character in self._visited_characters:
                chosen_character = random.choice(VALID_CHARACTERS)
            self._visited_characters.add(chosen_character)
            return chosen_character
