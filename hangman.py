"""Simple Hangman Game."""
from __future__ import annotations
from typing import Optional
import random
import time
import multiprocessing as mp

GAME_START_CHARACTER = '*'
VALID_CHARACTERS = 'abcdefghijklmnopqrstuvwxyz'


class Hangman:
    """A class representing a state of a game of Hangman.

    Public instance attributes:
        - total_tries: an int representing total tries allowed in a game.

    Private instance attributes:
        - _chosen_word: a list representing the chosen word,
        either by random or by the Player. This contains a sequence of str of each character.
        - _guess_status: a list representing current guessed status.
        - _count: an int representing how many guesses have been played.
        - _user_chosen_character: a str representing user's chosen character as a guess.
        - tries_left: an int representing the number of tries left before it's Game Over.
        By default there are 10 tries.

    Representation invariant:
        - len(Hangman._guessed_status) == len(Hangman._chosen_word)
    """
    total_tries: int = 10

    _chosen_word: list[str] = None
    _guess_status: list[str] = None
    _count: int = 0
    _user_guessed_character: str = None
    _tries_left: int = 10

    def __init__(self) -> None:
        """Initializes the Hangman variable."""

    def get_chosen_word(self) -> str:
        """Return the chosen word as a string.

        If the word has not yet been initialized, return an empty string.
        """
        if self.word_is_empty():
            return ''
        else:
            return ''.join(self._chosen_word)

    def get_guess_status(self) -> str:
        """Return the guess status as a string.

        If the guess status has not yet been initialized, return an empty string.
        """
        if self.guess_status_is_empty():
            return ''
        else:
            return ''.join(self._guess_status)

    def _choose_random_word(self) -> Optional[str]:
        """Return a random english word, in lower case."""
        # r = RandomWords()
        # return r.get_random_word(minLength=8).lower()
        with open('word_bank.txt') as file:
            r = random.randint(0, int(file.readline()) - 1)
            return file.readlines()[r].strip('\n')

    def set_tries(self, tries: int) -> None:
        """Set a new number of tries (mutates self._tries_left and self._total_tries)."""
        self._tries_left = tries
        self.total_tries = tries

    def is_valid_word(self, word: Optional[str]) -> bool:
        """Return whether the given word is a valid word.

        A word is valid when every character is in VALID_CHARACTERS.
        """
        if word is None:
            return False
        else:
            return all({char in VALID_CHARACTERS for char in word})

    def set_word(self, word: str = None) -> None:
        """Initializes the instance variable _chosen_word with the given word parameter.

        If word parameter is None, initializes the instance variable with a random word
        using the random-word library.
        """
        if word is not None:
            self._chosen_word = [char.lower() for char in word]
        else:
            random_word = None
            while not self.is_valid_word(random_word):
                random_word = self._choose_random_word()
            self._chosen_word = [char for char in random_word]

    def set_guess_status(self) -> None:
        """Initializes the instance variable _guess_status to a list of str.

        Before the game begins, it should be initialized as:
        ['?', '?', '?', ...]
        """
        if self.word_is_empty():
            raise EmptyWordError('Word not yet initialized')
        else:
            length = len(self._chosen_word)
            self._guess_status = ['?' for _ in range(length)]

    def get_num_tries(self) -> int:
        """Return the number of tries left."""
        return self._tries_left

    def get_efficiency_score(self, bonus_weight: float = 0.2) -> float:
        """Return the efficiency score of the player.

        The efficiency is a decreasing function from N to [1, 0)
        on the number of guesses made."""
        if self._count == 0:
            return 1
        distinct = len(set(self._chosen_word))

        return bonus_weight * max(0.0, 1 - self._count / distinct) + \
            (1 - bonus_weight) * min(1.0, distinct / self._count)

    def word_is_empty(self) -> bool:
        """Return whether the chosen_word is None (i.e., empty and not initialized)."""
        return self._chosen_word is None

    def guess_status_is_empty(self) -> bool:
        """Return whether the guessed_status is None (i.e., empty and not initialized)."""
        return self._guess_status is None

    def game_is_finished(self) -> bool:
        """Return whether the current game has ended.

        The hangman game is ended when there are 0 tries left,
        the user has completed the word, or
        the user has correctly guessed the word.
        """
        if self._tries_left == 0 or self._chosen_word == self._guess_status:
            return True
        else:
            return False

    def guess_character(self, character: str) -> None:
        """Guess a character using the given character.

        Preconditions:
            - len(character) == 1
            - character in VALID_CHARACTERS
        """
        if self.word_is_empty():
            raise EmptyWordError('Word not yet initialized')
        elif self.guess_status_is_empty():
            raise EmptyWordError('Guess status not yet initialized')

        self._count += 1

        # assert not self.guess_status_is_empty() and not self.word_is_empty()
        if character in self._chosen_word:
            index_list = []
            current_index = 0
            for char in self._chosen_word:
                if char == character:
                    index_list.append(current_index)
                current_index += 1
            # assert index_list != []
            current_index = 0
            for _ in self._guess_status:
                if current_index in index_list:
                    self._guess_status[current_index] = character
                current_index += 1

        else:
            self._tries_left -= 1

    def guess_word(self, word: str) -> None:
        """Guess a word using the given word.

        The given parameter word must only contain lower English alphabets.

        Preconditions:
            - len(word) > 1
            - all({char in VALID_CHARACTERS for char in word})
        """
        if self.word_is_empty():
            raise EmptyWordError('Word not yet initialized')
        elif self.guess_status_is_empty():
            raise EmptyWordError('Guess status not yet initialized')

        self._count += 1

        if self.get_chosen_word() == word:
            self._guess_status = self._chosen_word
        else:
            self._tries_left -= 1

    def make_guess(self, input_: str) -> None:
        """Makes a guess with the given input.

        Preconditions:
            - [char.lower() in VALID_CHARACTERS for char in input_]
        """
        if len(input_) == 1:
            self.guess_character(input_)
        else:
            self.guess_word(input_)


class EmptyWordError(Exception):
    """Raised when a guess is attempted on an empty word
    (i.e., a word that has not yet been initialized)."""


class Player:
    """An abstract class representing a Hangman AI.

    This class can be subclassed to implement different strategies for playing Hangman.

    can_guess_word determines whether the given AI Player can guess the full word.
    """
    can_guess_word: bool = False

    # Private instance attribute
    #   - : a set representing the characters (or words) that have already been guessed.
    #       No AI should repeat the guesses; each AI Player will use this instance variable
    #       to exclude the duplicate guesses.
    _visited_characters: set = set()

    def make_guess(self, game: Hangman, previous_guess: Optional[str]) -> str:
        """Make a guess given the current game.

        previous_guess is the player's most recently guessed character, or None if no guesses
        have been made.
        """
        raise NotImplementedError

    def clear_visited(self) -> None:
        """Clears the visited characters set"""
        self._visited_characters = set()


# Functions for running games

def run_games(n: int, player: Player) -> tuple[float, int, int, float]:
    """Run n Hangman games.

    Return a tuple containing the efficiency score, number of games won,
    total number of guesses, and time taken in seconds.
    """
    eff = 0
    won = 0
    guesses = 0
    start = time.perf_counter()
    for i in range(n):
        player.clear_visited()
        result = run_game(player)
        eff += result[0] * result[1]
        won += result[1]
        guesses += len(result[2]) - 1

    eff /= max(1, won)
    t = time.perf_counter() - start
    return (eff, won, guesses, t)


def run_games_async(n: int, player: Player, q: mp.Queue) -> None:
    """Run n Hangman games and report the results through the Queue"""
    result = run_games(n, player)
    q.put(result)


def run_game(player: Player, word: str = None,
             verbose: bool = False) -> \
        tuple[float, bool, list[str], str]:
    """Run a Hangman game.

    Return a tuple containing the efficiency score, a bool representing whether
    the player has won the game, a list of strings representing the sequence of
    all guesses made, and the correct word, in this order.

    If word is None, then the hangman API will choose a random word of length 8 or more.

    A gentle reminder:
        - run_game(...)[0] is a float representing the efficiency score.
        - run_game(...)[1] is a bool representing whether the player has won the game
        - run_game(...)[2] is a list of strings representing the sequence of all guesses made.
        - run_game(...)[3] is a string representing the correct word.
    """
    hangman = Hangman()
    hangman.set_word(word)
    hangman.set_guess_status()

    previous_character = None

    # For the purpose of inserting this sequence to the GameTree sequence,
    # GAME_START_CHARACTER is necessary.
    # For the purpose of visualization, the first element should be ignored.
    guess_sequence = [GAME_START_CHARACTER]
    assert hangman.get_chosen_word() != ""
    _correct_word = hangman.get_chosen_word()

    while not hangman.game_is_finished():
        user_guess = None
        while not hangman.is_valid_word(user_guess):
            user_guess = player.make_guess(hangman, previous_character)

        hangman.make_guess(user_guess)
        if verbose:
            print(user_guess, ' ', hangman.get_guess_status())

        previous_character = user_guess
        guess_sequence.append(user_guess)

    return (
        hangman.get_efficiency_score(),
        hangman.get_num_tries() > 0,
        guess_sequence,
        _correct_word
        )


def _extract_valid_words(file_path: str) -> None:
    """Write a file containing only the valid words.

    In this Hangman game, a word is valid iff all the characters in such word are in
    VALID_CHARACTERS.

    This function is PRIVATE, and it is only meant to be used once with the
    anagram_dictionary.txt file.
    """
    f = open("valid_words_large.txt", 'w+')
    with open(file_path) as file:
        for row in file:
            stripped = row.strip()
            is_valid = True
            for char in stripped:
                if char not in VALID_CHARACTERS:
                    is_valid = False
            if is_valid:
                f.write(row)
    f.close()


def run_example_game(word: str = None) -> None:
    """Run an example Hangman game."""
    hangman = Hangman()
    hangman.set_word(word)
    hangman.set_guess_status()

    print(hangman.get_guess_status())
    # For debug purposes, uncomment the following line:
    print('Correct word: ' + hangman.get_chosen_word())

    while True:
        user_guess = None
        while not hangman.is_valid_word(user_guess):
            user_guess = input('Make a guess: ')
            if not hangman.is_valid_word(user_guess.lower()):
                print('Invalid guess. Please try again.')

        hangman.make_guess(user_guess)
        print(hangman.get_guess_status())
        print('You have ' + str(hangman.get_num_tries()) + ' tries left!')
        if hangman.game_is_finished():
            break

    if hangman.get_num_tries() == 0:
        print('Game Over! You Lost :(')
    else:
        print('You won! Congrats :)')
    print('The correct word was: ' + hangman.get_chosen_word())
    print('Efficiency score: ' + str(hangman.get_efficiency_score()))


if __name__ == "__main__":
    # run_example_game()
    import hm_players
    # graph = hm_players.load_word_bank('Small.txt')
    # player = hm_players.GraphNextPlayer(graph)
    # print('Running game with GraphNextPlayer')
    graph = hm_players.load_word_bank('Small.txt', 'prev')
    player = hm_players.GraphPrevPlayer(graph, True)
    print('Running game with GraphPrevPlayer')

    state = run_game(player, 'snowflakes', verbose=True)
    print('Won' if state[1] else 'Lost')
    print('Word:', state[3])

    # import doctest
    # doctest.testmod()
    #
    # import python_ta.contracts
    # python_ta.contracts.check_all_contracts()
    # python_ta.check_all(config={
    #     'extra-imports': ['random', 'hm_players'],
    #     'allowed-io': ['open', 'print'],
    #     'max-line-length': 100,
    #     'disable': ['E1136']
    # })
