"""Simple Hangman Game.

Installation instructions:

    - Upgrade PIP to the latest version
    - Settings -> Project -> Python Interpreter -> '+' -> Search "random-word" -> Install
      (pip install random-word)
    - Settings -> Project -> Python Interpreter -> '+' -> Search "PyYAML" -> Install
      (pip install pyyaml)
"""
from random_word import RandomWords
from typing import Optional

GAME_START_CHARACTER = '*'
VALID_CHARACTERS = 'abcdefghijklmnopqrstuvwxyz'


class Hangman:
    """A class representing a state of a game of Hangman.

    Public instance attributes:
        - TOTAL_TRIES: an int representing total tries allowed in a game.

    Private instance attributes:
        - _chosen_word: a list representing the chosen word,
        either by random or by the Player. This contains a sequence of str of each character.
        - _guess_status: a list representing current guessed status.
        - _count: an int representing how many guesses have been played.
        - _user_chosen_character: a str representing user's chosen character as a guess.
        - tries_left: an int representing the number of tries left before it's Game Over.
        By default there are 10 tries.
        - _efficiency_score: a float representing how efficient an AI / a user is.
        The default score is 100.0.

    Representation invariant:
        - len(Hangman._guessed_status) == len(Hangman._chosen_word)
    """
    TOTAL_TRIES = 10

    _chosen_word: list[str] = None
    _guess_status: list[str] = None
    _count: int = 0
    _user_guessed_character: str = None
    _tries_left: int = 10
    _efficiency_score: float = 100.0

    def __init__(self) -> None:
        """Initializes the Hangman variable."""
        pass

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
        r = RandomWords()
        return r.get_random_word(minLength=8).lower()

    def set_tries(self, tries: int) -> None:
        """Set a new number of tries (mutates self._tries_left and self._total_tries)."""
        self._tries_left = tries
        self.TOTAL_TRIES = tries

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

    def get_efficiency_score(self) -> float:
        """Return the efficiency score."""
        return self._efficiency_score

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
        else:
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

                self._efficiency_score -= 20 / self.TOTAL_TRIES
                if self._efficiency_score < 0:
                    self._efficiency_score = 0.0
            else:
                self._tries_left -= 1
                self._efficiency_score -= 100 / self.TOTAL_TRIES
                if self._efficiency_score < 0:
                    self._efficiency_score = 0.0

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
        else:
            if self.get_chosen_word() == word:
                self._guess_status = self._chosen_word
            else:
                self._tries_left -= 1
                self._efficiency_score -= 100 / self.TOTAL_TRIES
                if self._efficiency_score < 0:
                    self._efficiency_score = 0.0

    def make_guess(self, input_: str) -> None:
        """Makes a guess with the given input.

        Preconditions:
            - [char.lower() in VALID_CHARACTERS for char in input_]
        """
        if len(input_) == 1:
            self.guess_character(input_)
        else:
            self.guess_word(input_)

    # TODO: Make general run_game(s) methods
    #       for better usage


class EmptyWordError(Exception):
    """Raised when a guess is attempted on an empty word
    (i.e., a word that has not yet been initialized)."""
    pass


class Player:
    """An abstract class representing a Hangman AI.

    This class can be subclassed to implement different strategies for playing Hangman.
    """
    # Private instance attribute
    #   - : a set representing the characters that have already been guessed.
    #       No AI should repeat the guesses; each AI Player will use this instance variable
    #       to exclude the duplicate guesses.
    _visited_characters = set()

    def make_guess(self, game: Hangman, previous_character: Optional[str]) -> str:
        """Make a guess given the current game.

        previous_character is the player's most recently guessed character, or None if no guesses
        have been made.
        """
        raise NotImplementedError


################################################################################
# Functions for running games
################################################################################
DEFAULT_FPS = 6  # Default number of moves per second to display in the visualization


def run_games(n: int, white: Player, black: Player,
              visualize: bool = False, fps: int = DEFAULT_FPS,
              show_stats: bool = False) -> None:
    """...

        - TODO: Change the parameters; visualize, show_stats and fps are
                no longer necessary
    """
    # TODO: Fill in
    #       Call run_game and maybe use for loops?
    ...


def run_game(player: Player, word: str = None) -> tuple[float, bool, list[str], str]:
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

        previous_character = user_guess
        guess_sequence.append(user_guess)

    if hangman.get_num_tries() == 0:
        assert hangman.get_efficiency_score() == 0.0
        return (
            hangman.get_efficiency_score(),
            False,
            guess_sequence,
            _correct_word
        )
    else:
        return (
            hangman.get_efficiency_score(),
            True,
            guess_sequence,
            _correct_word
        )


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
    run_example_game()
