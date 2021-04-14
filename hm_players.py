"""All Hangman AI Players.

Brief explanation to each AI player:
    - RandomPlayer: only plays randomly
    - RandomGraphPlayer: plays randomly based on the given GameGraph
    - GraphNextPlayer: if applicable, guesses the next character in the given GameGraph
    - GraphPrevPlayer: if applicable, guesses the previous character in the given GameGraph
    - FrequentPlayer: only guesses the frequently guessed characters
"""
from __future__ import annotations
import random
from typing import Optional, Any
import enchant

from hm_game_graph import _WeightedVertex, GameGraph
import hangman

VALID_CHARACTERS = 'abcdefghijklmnopqrstuvwxyz'


def load_word_bank(games_file: str, order: str = 'next') -> GameGraph:
    """Create a word bank (i.e., a game graph) based on games_file.

    Preconditions:
        - order in {'next', 'prev', 'both'}
    """
    graph = GameGraph()
    with open(games_file) as file:
        for row in file:
            if order != 'prev':
                graph.insert_character_sequence(row.strip('\n'))
            if order != 'next':
                graph.insert_character_sequence(row.strip('\n')[::-1])

    return graph


class RandomPlayer(hangman.Player):
    """A Hangman AI whose strategy is always picking a random character.
    This AI will never guess a full word.
    """

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
    """A Hangman player that plays randomly based on a give GameGraph.

    This player uses a game graph to make guesses as the game is played.
    On its turn:
        - At first, the AI will choose randomly from all the available characters (i.e., vertices)
        in the given GameGraph.
        - If there are neighbours to the previously chosen vertex, choose a random one
        for the next guess
        - If there are no neighbours, the AI guesses randomly.
    This AI will never guess a full word.
    """
    # Private Instance Attributes:
    #   - _game_graph:
    #       The GameGraph that this player uses to make its guesses. If None, then this
    #       player just makes random guesses.
    _graph: Optional[GameGraph]

    def __init__(self, graph: GameGraph) -> None:
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
            return self._random_vertex_guess()
        else:
            # Not first guess, makes random guesses based on neighbours
            try:
                get_previous_vertex = self._graph.get_vertex_by_item(previous_character)
            except ValueError:
                random_item = random.choice(list(self._graph.get_all_vertices()))
                get_previous_vertex = self._graph.get_vertex_by_item(random_item)
            all_neighbouring_vertices = [v for v in get_previous_vertex.neighbours]
            if len(all_neighbouring_vertices) == 0:
                # If there are no neighbours (highly unlikely), guess random character
                return self._random_character_guess()
            else:
                # If there are at least one neighbour, guess randomly among its neighbours
                all_neighbouring_vertices_copy = all_neighbouring_vertices.copy()
                chosen_vertex = random.choice(all_neighbouring_vertices_copy)
                self._mutate_all_neighbouring_vertices(chosen_vertex,
                                                       all_neighbouring_vertices_copy)
                if (len(all_neighbouring_vertices_copy)) == 0:
                    return self._random_vertex_guess()
                else:
                    self._visited_characters.add(chosen_vertex.item)
                    return chosen_vertex.item

    # The following three private methods are to satisfy PythonTA; we have divided up the
    # make_guess method above with three helper methods to simplify large nested
    # if-else statements.

    def _random_vertex_guess(self) -> Any:
        """Return a random character among all the vertices."""
        all_items = self._graph.get_all_vertices()
        chosen_item = random.choice(list(all_items))
        while chosen_item in self._visited_characters:
            chosen_item = random.choice(list(all_items))
        self._visited_characters.add(chosen_item)
        return chosen_item

    def _random_character_guess(self) -> Any:
        """Return a purely random character from VALID_CHARACTERS."""
        all_characters_list = [char for char in VALID_CHARACTERS]
        random_character = random.choice(all_characters_list)
        while random_character in self._visited_characters:
            random_character = random.choice(all_characters_list)
        self._visited_characters.add(random_character)
        return random_character

    def _mutate_all_neighbouring_vertices(self, chosen_vertex: _WeightedVertex,
                                          all_neighbouring_vertices_copy:
                                          list[_WeightedVertex]) -> None:
        """Mutate the all_neighbouring_vertices_copy.
        Specifically, remove each vertex from this shallow copy until either the
        chosen_vertex contains the wanted (valid) item, or all the elements of
        all_neighbouring_vertices_copy have been removed.
        """
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
    _graph: Optional[GameGraph]
    _dict: Optional[enchant.Dict]

    def __init__(self, graph: GameGraph, can_guess_word: bool = False) -> None:
        """Initialize this player.

        Preconditions:
            - graph represents a game graph
        """
        self._graph = graph
        self._visited_characters = set()
        self._dict = enchant.Dict('en_US')
        self.can_guess_word = can_guess_word

    def make_guess(self, game: hangman.Hangman, previous_guess: Optional[str]) -> str:
        """Make a guess given the current game.

        previous_guess is the player's most recently guessed character, or None if no guesses
        have been made.
        """
        status = game.get_guess_status()
        if status == '?' * len(status):
            # Beginning, choose most common character
            # print('Beginning ', end='  ')
            return self.frequency_guess()

        # Guess adjacent character (first one seen)
        choice = self.adjacent_guess(game)
        if choice is not None:
            # print('Adjacent', choice[1], end='  ')
            return choice[0]

        # If most letters are known then guess entire word
        num_unknown = status.count('?')
        if self.can_guess_word and num_unknown <= 0.4 * len(status):
            result = self.guess_word(game)
            if result is not None:
                # print('Word      ', end='  ')
                return result

        # Last resort, random guess
        # print('Random    ', end='  ')
        return self.random_guess()

    def frequency_guess(self) -> str:
        """Guess the most common letter"""
        chars = {(w, self._graph.get_vertex_weight(w))
                 for w in self._graph.get_all_vertices()
                 if w not in self._visited_characters}
        choice = max(chars, key=lambda p: p[1])[0]
        self._visited_characters.add(choice)
        return choice

    def adjacent_guess(self, game: hangman.Hangman,
                       dry_run: bool = False) -> Optional[tuple[str, str, int]]:
        """Guess a letter that comes after a known letter.
        Returns (choice, s, index) where s is the known letter."""
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
                    if not dry_run:
                        self._visited_characters.add(choice)
                    return (choice, s, i)

    def random_guess(self) -> str:
        """Make a random guess"""
        char = random.choice(VALID_CHARACTERS)
        while char in self._visited_characters:
            char = random.choice(VALID_CHARACTERS)
        self._visited_characters.add(char)
        return char

    def guess_word(self, game: hangman.Hangman) -> Optional[str]:
        """Guess the entire word"""
        status = game.get_guess_status()
        current_word = status.replace('?', '')
        visited_character = self._visited_characters
        suggested_word = _suggest_valid_word(current_word, visited_character, game, self._dict)
        if suggested_word == '':
            return None
        else:
            self._visited_characters.add(suggested_word)
            return suggested_word


class GraphPrevPlayer(GraphNextPlayer):
    """This player is the counterpart to GraphNextPlayer.
    It guesses the letter before a known letter."""

    def adjacent_guess(self, game: hangman.Hangman,
                       dry_run: bool = False) -> Optional[tuple[str, str, int]]:
        """Guess the letter that comes before a known letter.
        Returns (choice, s, index) where s is the known letter."""
        status = game.get_guess_status()
        for i in range(len(status) - 1):
            s = status[i + 1]
            n = status[i]
            if (s in VALID_CHARACTERS) and (n == '?') and (s in self._graph):
                chars = {(w, self._graph.get_weight(s, w))
                         for w in self._graph.get_neighbours(s)
                         if w not in self._visited_characters}
                if len(chars) > 0:
                    choice = max(chars, key=lambda p: p[1])[0]
                    if not dry_run:
                        self._visited_characters.add(choice)
                    return (choice, s, i)


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
    _graph: Optional[GameGraph]
    _dict: Optional[enchant.Dict]

    def __init__(self, graph: GameGraph, can_guess_word: bool = False) -> None:
        """Initialize this player.

        Preconditions:
            - graph represents a game graph
        """
        self._graph = graph
        self._visited_characters = set()
        self._dict = enchant.Dict('en_US')
        self.can_guess_word = can_guess_word

    def make_guess(self, game: hangman.Hangman, previous_guess: Optional[str]) -> str:
        """Make a guess given the current game.

        previous_guess is the player's most recently guessed character, or None if no guesses
        have been made.
        """
        guess_status = game.get_guess_status()
        num_unknown = guess_status.count('?')
        # The AI only attempts to guess the full word if self.can_guess_word is True
        # and the number of unknown characters (i.e., '?') is less than or equal to
        # 40% of the number of total characters.
        if self.can_guess_word and num_unknown <= 0.4 * len(guess_status):
            current_word = guess_status.replace('?', '')
            visited_character = self._visited_characters
            suggested_word = _suggest_valid_word(current_word, visited_character, game, self._dict)
            if suggested_word == '':
                return self._make_guess_body()
            else:
                # print('guessed the full word')
                self._visited_characters.add(suggested_word)
                return suggested_word
        else:
            return self._make_guess_body()

    def _make_guess_body(self) -> str:
        """The main function body for the make_guess method above to simplify the code."""
        all_items = self._graph.get_all_vertices()
        max_weight = -1
        max_item = None
        for item in all_items:
            weight = self._graph.get_vertex_weight(item)
            if weight > max_weight:
                if item not in self._visited_characters:
                    max_weight = weight
                    max_item = item
        self._visited_characters.add(max_item)
        return max_item


def _suggest_valid_word(current_word: str, visited_characters: set,
                        game: hangman.Hangman, word_dict: enchant.Dict) -> str:
    """Return a suggested word based on the current_word (which should be incomplete /
    misspelled). Return an empty string if there are no valid suggestions.

    The returned word will NOT be in the player._visited_character set.
    """
    guess_status_sequence = [char for char in game.get_guess_status()]
    given_word_length = len(guess_status_sequence)
    chosen_word = ''
    all_suggested_words = word_dict.suggest(current_word)
    for word in all_suggested_words:
        if not game.is_valid_word(word):
            continue
        suggested_sequence = [char for char in word]
        if word not in visited_characters and len(word) == given_word_length and \
                _is_valid_sequence(suggested_sequence, guess_status_sequence):
            chosen_word = word
            break
    return chosen_word


def _is_valid_sequence(suggested_sequence: list[str], current_sequence: list[str]) -> bool:
    """Return whether suggested_sequence is a valid character sequence.

    A character sequence is valid when all characters in suggested_sequence exactly match the
    corresponding characters in current_sequence at the given indices, or such characters in
    current_sequence are '?'.

    Preconditions:
        - len(suggested_sequence) == len(current_sequence)
        - all({isinstance(char, str) for char in current_sequence})
        - all({isinstance(char, str) for char in suggested_sequence})
    """
    for i in range(len(suggested_sequence)):
        if current_sequence[i] == '?':
            continue
        elif current_sequence[i] != suggested_sequence[i]:
            return False
    return True


if __name__ == "__main__":
    g = load_word_bank('valid_words_large.txt')
    h = hangman.Hangman()

    import time
    start = time.perf_counter()
    count = 0

    # random_p = RandomPlayer()
    # for _ in range(100):
    #     print(hangman.run_game(random_p))
    #     random_p._visited_characters = set()

    # frequent_p = FrequentPlayer(g, can_guess_word=True)
    # for _ in range(100):
    #     print(hangman.run_game(frequent_p))
    #     frequent_p._visited_characters = set()
    #     count += 1
    #     if time.perf_counter() - start > 10:
    #         print(count)
    #         break
    # random_graph_p = RandomGraphPlayer(g)
    # for _ in range(100):
    #     print(hangman.run_game(random_graph_p))
    #     random_graph_p._visited_characters = set()

    # graph_next_p = GraphNextPlayer(g, can_guess_word=True)
    # for _ in range(100):
    #     # TODO: if testing GraphNextPlayer with this code,
    #     #       comment out 4 print statements inside GraphNextPlayer.make_guess()
    #     print(hangman.run_game(graph_next_p))
    #     graph_next_p._visited_characters = set()

    # graph_prev_p = GraphPrevPlayer(g, can_guess_word=True)
    # for _ in range(100):
    #     # TODO: if testing graph_prev_p with this code,
    #     #       comment out 4 print statements inside GraphNextPlayer.make_guess()
    #     print(hangman.run_game(graph_prev_p))
    #     graph_prev_p._visited_characters = set()

    # import doctest
    # doctest.testmod()
    #
    # import python_ta.contracts
    # python_ta.contracts.check_all_contracts()
    # python_ta.check_all(config={
    #     'extra-imports': ['hm_game_graph', 'hangman', 'random'],
    #     'allowed-io': ['open', 'print'],
    #     'max-line-length': 100,
    #     'disable': ['E1136']
    # })
