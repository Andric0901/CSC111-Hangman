"""All Hangman AI Players."""

import random
from typing import Optional

import hm_game_tree
import hm_game_graph
import hangman

VALID_CHARACTERS = 'abcdefghijklmnopqrstuvwxyz'


def load_word_bank(games_file: str, order: str = 'next') -> hm_game_graph.GameGraph:
    """Create a word bank (i.e., a game graph) based on games_file.

    Preconditions:
        - order in {'next', 'prev', 'both'}
    """
    graph = hm_game_graph.GameGraph()
    with open(games_file) as file:
        for row in file:
            if order != 'prev':
                graph.insert_character_sequence(row.strip('\n'))
            if order != 'next':
                graph.insert_character_sequence(row.strip('\n')[::-1])

    return graph


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


class RandomGraphPlayer(hangman.Player):
    """A hangman player that plays randomly based on a give GameGraph.

    This player uses a game graph to make guesses as the game is played.
    On its turn:
        - At first, the AI will choose randomly from all the available characters (i.e., vertices)
        in the given GameGraph.
        - If there are neighbours to the previously chosen vertex, choose a random one
        for the next guess
        - If there are no neighbours, the AI guesses randomly.
    """
    # Private Instance Attributes:
    #   - _game_graph:
    #       The GameGraph that this player uses to make its guesses. If None, then this
    #       player just makes random guesses.
    _graph: Optional[hm_game_graph.GameGraph]

    def __init__(self, graph: hm_game_graph.GameGraph) -> None:
        """Initialize this player.

        Preconditions:
            - graph represents a game graph
        """
        self._graph = graph
        self._visited_characters = set()

    def make_guess(self, game: hangman.Hangman, previous_character: Optional[str]) -> str:
        """Make a guess given the current game.

        previous_guess is the player's most recently guessed character, or None if no guesses
        have been made.
        """
        if previous_character is None:
            # First guess, choose randomly among all the vertices
            all_items = self._graph.get_all_vertices()
            chosen_item = random.choice(list(all_items))
            while chosen_item in self._visited_characters:
                chosen_item = random.choice(list(all_items))
            self._visited_characters.add(chosen_item)
            return chosen_item
        else:
            # Not first guess, makes random guesses based on neighbours
            get_previous_vertex = self._graph.get_vertex_by_item(previous_character)
            all_neighbouring_vertices = [v for v in get_previous_vertex.neighbours]
            if len(all_neighbouring_vertices) == 0:
                # If there are no neighbours (highly unlikely), guess randomly
                all_characters_list = [char for char in VALID_CHARACTERS]
                random_character = random.choice(all_characters_list)
                while random_character in self._visited_characters:
                    random_character = random.choice(all_characters_list)
                self._visited_characters.add(random_character)
                return random_character
            else:
                # If there are at least one neighbour, guess randomly among its neighbours
                all_neighbouring_vertices_copy = all_neighbouring_vertices.copy()
                chosen_vertex = random.choice(all_neighbouring_vertices_copy)
                while True:
                    if chosen_vertex.item not in self._visited_characters or \
                            len(all_neighbouring_vertices_copy) == 0:
                        break
                    else:
                        all_neighbouring_vertices_copy.remove(chosen_vertex)
                        if len(all_neighbouring_vertices_copy) == 0:
                            break
                        else:
                            chosen_vertex = random.choice(all_neighbouring_vertices_copy)

                if (len(all_neighbouring_vertices_copy)) == 0:
                    all_items = self._graph.get_all_vertices()
                    chosen_item = random.choice(list(all_items))
                    while chosen_item in self._visited_characters:
                        chosen_item = random.choice(list(all_items))
                    self._visited_characters.add(chosen_item)
                    return chosen_item
                else:
                    self._visited_characters.add(chosen_vertex.item)
                    return chosen_vertex.item


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
    #       The GameGraph that this player uses to make its guesses. If None, then this
    #       player just makes random guesses.
    _graph: Optional[hm_game_graph.GameGraph]

    def __init__(self, graph: hm_game_graph.GameGraph) -> None:
        """Initialize this player.

        Preconditions:
            - graph represents a game graph
        """
        self._graph = graph
        self._visited_characters = set()

    def make_guess(self, game: hangman.Hangman, previous_guess: Optional[str]) -> str:
        """Make a guess given the current game.

        previous_guess is the player's most recently guessed character, or None if no guesses
        have been made.
        """
        status = game.get_guess_status()
        if status == '?' * len(status):
            # Beginning, choose most common character
            print('Beginning ', end='  ')
            return self.frequency_guess(game)

        # Guess adjacent character (first one seen)
        choice = self.next_guess(game)
        if choice is not None:
            print('Adjacent', choice[1], end='  ')
            return choice[0]

        # Last resort, random guess
        print('Random    ', end='  ')
        return self.random_guess()

    def frequency_guess(self, game: hangman.Hangman) -> str:
        """Guess the most common letter"""
        chars = {(w, self._graph.get_vertex_weight(w))
                 for w in self._graph.get_all_vertices()
                 if w not in self._visited_characters}
        choice = max(chars, key=lambda p: p[1])[0]
        self._visited_characters.add(choice)
        return choice

    def next_guess(self, game: hangman.Hangman) -> Optional[tuple[str, str]]:
        """Guess a letter that comes after a known letter.
        Returns (choice, s) where s is the known letter."""
        status = game.get_guess_status()
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
                    return (choice, s)

    def random_guess(self) -> str:
        """Make a random guess"""
        char = random.choice(VALID_CHARACTERS)
        while char in self._visited_characters:
            char = random.choice(VALID_CHARACTERS)
        self._visited_characters.add(char)
        return char


class FrequentPlayer(hangman.Player):
    """A Hangman player that only guesses the frequently guessed characters.

    This player uses a game graph to make guesses as the game is played.
    On its turn, it chooses the most frequently guessed character
    that has not been guessed before.
    """
    # Private Instance Attributes:
    #   - _game_graph:
    #       The GameGraph that this player uses to make its guesses. If None, then this
    #       player just makes random guesses.
    _graph: Optional[hm_game_graph.GameGraph]

    def __init__(self, graph: hm_game_graph.GameGraph) -> None:
        """Initialize this player.

        Preconditions:
            - graph represents a game graph
        """
        self._graph = graph
        self._visited_characters = set()

    def make_guess(self, game: hangman.Hangman, previous_guess: Optional[str]) -> str:
        """Make a guess given the current game.

        previous_guess is the player's most recently guessed character, or None if no guesses
        have been made.
        """
        all_items = self._graph.get_all_vertices()
        max_weight = -1
        max_item = None
        for item in all_items:
            weight = self._graph.get_vertex_weight(item)
            if weight > max_weight:
                if item not in self._visited_characters:
                    max_weight = weight
                    max_item = item
        if max_item is None:
            # TODO: All items are guessed... Does this case ever happen?
            ...
        else:
            # print((max_item, max_weight))
            self._visited_characters.add(max_item)
            return max_item


# TODO: Not necessary anymore
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


if __name__ == "__main__":
    import hangman
    g = load_word_bank('valid_words_large.txt')
    h = hangman.Hangman()

    # random_p = RandomPlayer()
    # for _ in range(500000):
    #     print(hangman.run_game(random_p))
    #     random_p._visited_characters = set()

    # frequent_p = FrequentPlayer(g)
    # for _ in range(500000):
    #     print(hangman.run_game(frequent_p))
    #     frequent_p._visited_characters = set()

    # random_graph_p = RandomGraphPlayer(g)
    # for _ in range(500000):
    #     print(hangman.run_game(random_graph_p))
    #     random_graph_p._visited_characters = set()

    # graph_next_p = GraphNextPlayer(g)
    # for _ in range(500000):
    #     # TODO: if testing GraphNextPlayer with this code,
    #     #       comment out 3 print statements inside GraphNextPlayer.make_guess()
    #     print(hangman.run_game(graph_next_p))
    #     graph_next_p._visited_characters = set()
