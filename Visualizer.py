"""An interactive tkinter window.

    - Install/Update PIL (In the case of denied access, install Pillow instead)
    - Install numpy
    - Install tkinter
"""

# Main menu + AI selection + Visualization

from tkinter import Frame, Tk, Canvas, N, E, S, W
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np
import time
from math import sin, cos, pi

import hangman
import hm_players
import hm_game_graph

f = ('Times', 32, 'bold')


class Coords:
    """Represents coordinates for an element in the GUI

    Instance attributes:
        - pos: The element is centered on these (x, y) coordinates.
        - w2, h2: The radii of the element (1/2 of its dimensions).
        - bounds: The (left, up, right, down) coordinates.
    """
    pos: tuple[int, int]
    w2: int
    h2: int
    bounds: tuple[int, int, int, int]

    def __init__(self, x: int, y: int, w2: int, h2: int) -> None:
        self.pos = (x, y)
        self.w2 = w2
        self.h2 = h2
        self.bounds = (x - w2, y - h2, x + w2, y + h2)


class Project(Frame):
    """Tkinter window for the main menu"""

    def __init__(self, root=None) -> None:
        """Initializes the window"""

        if root is None:
            root = Tk()
        super().__init__(root)

        self.root = root
        self.root.title('CSC111 Project')
        self.W = 960
        self.H = 600

        self.totFrames = 0
        self.startTime = time.time()

        # Center window
        offsetX = root.winfo_screenwidth()//2 - self.W//2
        offsetY = root.winfo_screenheight()//2 - self.H//2
        root.geometry("+{}+{}".format(offsetX, offsetY))
        root.lift()

        self.canvasItems = []

        self.window = 'Menu'

        self.loadMenuAssets()

    def loadMenuAssets(self) -> None:
        """Opens image assets used for main menu"""
        bg = Image.open('Assets/Background.jpg').convert('RGBA')
        self.background = np.array(bg, 'float32')

        self.title = np.array(Image.open('Assets/Title.png'), 'float32')

        graphic = Image.open('Assets/Hangman.png').resize((180, 150))
        self.graphic = 255 - np.array(graphic, 'float32')

        light = Image.open('Assets/LightRay.png')
        self.light = np.array(light, 'float32') * 0.6

        letterImgs = {'N': 0, 'V': 0, 'H': 0, 'P': 0}
        letterFile = 'Assets/Letters{}.png'
        for i in letterImgs:
            letterImgs[i] = np.array(Image.open(letterFile.format(i)), 'float32')
            letterImgs[i] *= np.expand_dims(letterImgs[i][:,:,3] / 255, -1)
        self.letterImgs = letterImgs

        self.lettersOffsets = ((np.random.random((8, 2)) - 0.5) * 400).astype('int')

        self.names = np.array(Image.open('Assets/Names.png'), 'float32')

        self.button = np.array(Image.open('Assets/Button.png'), 'float32')

        # Copy so we can change the intensity of each button individually
        self.buttons = [np.array(self.button),
                        np.array(self.button),
                        np.array(self.button),
                        np.array(self.button)]

        dims = (120, 60)
        self.buttonPos = [
            Coords(self.W*2//9, self.H-360, *dims),
            Coords(self.W*2//9, self.H-180, *dims),
            Coords(self.W*7//9, self.H-360, *dims),
            Coords(self.W*7//9, self.H-180, *dims)
            ]

        self.cursor = np.array(Image.open('Assets/Cursor.png'), 'float32')

    def loadSelectionAssets(self) -> None:
        """Opens image assets used in AI selection menu"""
        panel = np.array(Image.open('Assets/Panel.png'), 'float32')
        title = np.array(Image.open('Assets/Title_select.png'), 'float32')
        buttonImg = Image.open('Assets/Button2.png').resize((243, 61))
        button = np.array(buttonImg, 'float32')
        rect = np.array(Image.open('Assets/Rectangle.png'), 'float32')
        rect[:,:,:3] = 0
        rect[:,:,3] *= 1.2

        bg = np.array(self.background)
        self.blend(bg, title, (self.W//2, self.H//8-40), 'add')
        self.blend(bg, self.light, (self.W//2-80, self.H//2), 'add')
        self.blend(bg, self.light, (self.W//2+50, self.H//2), 'add')
        self.blend(bg, panel, (self.W//2, self.H//2+25), 'alpha')
        self.blend(bg, rect, (self.W*2//3 - 20, self.H*2//5 + 10), 'alpha')
        self.temp_bg = np.clip(bg, 0, 255)

        # Number of buttons / player types
        NB = 7

        self.button = np.array(button)
        self.buttons = [np.array(button) for _ in range(NB)]

        symbols = np.array(Image.open('Assets/Symbols.png'), 'float32')
        self.symbolImg = symbols
        self.symbols = [
            symbols[60*i:60*(i+1)] for i in range(NB)
            ]

        dims = (120, 32)
        self.buttonPos = [
            Coords(self.W*2//7-10, 118 + 69 * i, *dims)
            for i in range(NB)
            ]

        self.selectedButton = None
        self.can_guess_word = True

        self.button3 = np.array(Image.open('Assets/Button3.png').resize((144, 48)),
                                'float32')
        self.wordButton = Coords(self.W*3//4 + 10, self.H*3//4 + 10, 72, 24)

    def loadVisualizeAssets(self) -> None:
        """Opens image assets used in visualization screen"""
        panel = np.array(Image.open('Assets/Panel.png'), 'float32')
        title = np.array(Image.open('Assets/Title_visualize.png'), 'float32')
        rects = np.array(Image.open('Assets/Rectangles.png'), 'float32')
        rects[:,:,:3] = 0

        bg = np.array(self.background)
        self.blend(bg, title, (self.W//2, self.H//8-40), 'add')
        self.blend(bg, self.light, (self.W//2-80, self.H//2), 'add')
        self.blend(bg, self.light, (self.W//2+50, self.H//2), 'add')
        self.blend(bg, panel, (self.W//2, self.H//2+25), 'alpha')
        self.blend(bg, rects, (self.W//2 - 20, self.H//2 + 30), 'alpha')

        PAD = 8
        padded = np.zeros((self.graphic.shape[0] + PAD * 2,
                           self.graphic.shape[1] + PAD * 2, 4), 'float32')
        padded[PAD:-PAD, PAD:-PAD] = self.graphic
        graphic = Image.fromarray(padded.astype('uint8'))
        graphicBlur = graphic.filter(ImageFilter.GaussianBlur(8))
        self.graphicBlur = np.array(graphicBlur, 'float32')

        self.graphic = 255 - self.graphic
        self.graphic[:,:,3] = 255 - self.graphic[:,:,0]

        self.charBox = np.array(Image.open('Assets/Character.png'), 'float32')
        self.charLight = np.array(Image.open('Assets/Highlight.png'), 'float32')
        self.badge = np.array(Image.open('Assets/Badge.png'), 'float32')

        playerSym = self.symbols[self.selectedButton[0]]
        self.blend(bg, self.badge, (self.W//6 - 5, self.H//6 + 20), 'alpha')
        self.blend(bg, playerSym, (self.W//6 - 5, self.H//6 + 20), 'alpha')

        self.temp_bg = np.clip(bg, 0, 255)

        buttonImg = Image.open('Assets/Button3.png').resize((166, 48))
        self.button = np.array(buttonImg, 'float32')

        dims = (83, 24)
        self.buttons = [np.array(self.button) for _ in range(4)]
        self.buttonPos = [
            Coords(self.W//4, self.H//3 - 25, *dims),
            Coords(self.W//4, self.H//3 + 35, *dims),
            Coords(self.W*4//5 - 20, self.H*3//5 + 5, *dims),
            Coords(self.W*3//4 - 20, self.H*5//6 + 30, *dims)
            ]

        self.incFFButton = Coords(self.W*5//6 + 10, self.H*3//5 - 10, 20, 20)
        self.decFFButton = Coords(self.W*5//6 + 10, self.H*3//5 + 20, 20, 20)

        self.autoPlay = False
        self.playerGraph = None
        self.numFFGames = 100
        self.statText = ''
        self.startGame()

    def start(self) -> None:
        """Start the rendering loop"""
        self.makeWidgets()
        self.renderMenu()
        self.after(10, self.updateCanvas)

    def makeWidgets(self) -> None:
        """Create the canvas to draw on"""
        self.grid(sticky=N+E+S+W)

        self.d = Canvas(self, width=self.W, height=self.H,
                        highlightthickness=0, highlightbackground='black')
        self.d.grid(row=0, column=0, sticky=N+E+S+W)
        self.d.config(background='#000', cursor='none')
        self.d.bind('<Button-1>', self.clicked)
        self.finalRender = self.d.create_image((self.W/2, self.H/2))

    def renderMenu(self) -> None:
        """Render the main menu"""
        self.totFrames += 1

        # Copy background
        frame = np.array(self.background)

        # Blend in title and names
        self.blend(frame, self.title, (self.W//2, self.H//8), 'add')
        self.blend(frame, self.graphic, (self.W//2, self.H//2), 'add')
        self.blend(frame, self.names, (self.W//2, self.H - 60), 'add')

        # Blend in decorative text
        freq =  (11, 17, 23, 19,  26, 13, 12, 16)
        start = (-8, 22,  0, -15, 18, -3, 16, -4)
        pos = self.lettersOffsets
        abc = 'NVHP' * 2
        for i in range(len(abc)):
            intensity = max(0, sin((self.totFrames + start[i]) / freq[i]))
            if intensity == 0:
                continue
            img = self.letterImgs[abc[i]] * intensity

            position = (self.W//2 + pos[i][0], self.H//2 + pos[i][1])
            self.blend(frame, img, position, 'add')

        w1 = self.W//2 + int(sin(self.totFrames / 29) * 100) - int(sin(self.totFrames / 17) * 20)
        w2 = self.W//2 - int(sin(self.totFrames / 29) * 100) + int(sin(self.totFrames / 17) * 20)
        self.blend(frame, self.light, (w1, self.H//2), 'add')
        self.blend(frame, self.light, (w2, self.H//2), 'add')

        # Blend in menu buttons
        for i in range(len(self.buttons)):
            self.blend(frame, self.buttons[i], self.buttonPos[i].pos, 'alpha')

        self.blendCursor(frame)
        self.displayImage(frame)
        self.clearCanvas()

        # Add text to buttons
        texts = ['Instructions', 'Play!', 'Compare AI', 'Quit']
        self.texts = [self.d.create_text(*self.buttonPos[i].pos,
                                         text=texts[i], fill='#fff', font=f)
                      for i in range(len(self.buttons))]

        self.canvasItems = self.texts

    def renderSelect(self) -> None:
        """Render the AI selection screen"""
        self.totFrames += 1

        frame = np.array(self.temp_bg)

        for i in range(len(self.buttons)):
            px, py = self.buttonPos[i].pos
            self.blend(frame, self.buttons[i], (px, py), 'alpha')
            self.blend(frame, self.symbols[i], (px - 92, py), 'alpha')

        self.blend(frame, self.button3, self.wordButton.pos, 'alpha')

        self.blendCursor(frame)
        self.displayImage(frame)
        self.clearCanvas()

        texts = ['RandomPlayer', 'RandomGraphPlayer', 'GraphNextPlayer',
                 'GraphPrevPlayer', 'GraphAdjPlayer',
                 'FrequentPlayer', 'Human Player']
        self.texts = [self.d.create_text(self.buttonPos[i].pos[0] - 58,
                                         self.buttonPos[i].pos[1],
                                         text=texts[i], fill='#fff',
                                         anchor=W,
                                         font=('Times', 15 if i == 1 else 17))
                      for i in range(len(self.buttons))]

        # Use the docstrings as description
        x = self.d.winfo_pointerx() - self.d.winfo_rootx()
        y = self.d.winfo_pointery() - self.d.winfo_rooty()

        for i in range(len(self.buttons)):
            if self.selected(x, y, self.buttonPos[i].bounds):
                self.selectedButton = (i, texts[i])

        if self.selectedButton is not None:
            if self.selectedButton[0] == 6:
                t = 'Human player: You!'
            else:
                t = getattr(hm_players, self.selectedButton[1]).__doc__
            self.texts.append(
                self.d.create_text(
                    self.W//2 - 20, self.H//4 - 25,
                    text=t, fill='#fff', anchor=N + W,
                    font=('Times', 11), width=340
                    )
                )

        self.texts.append(
            self.d.create_text(
                self.W//2 - 30, self.H*3//4 + 10,
                text='Can guess entire words:', fill='#000',
                anchor=W, font=('Times', 15)
                )
            )
        self.texts.append(
            self.d.create_text(
                *self.wordButton.pos,
                text=str(self.can_guess_word), fill='#fff',
                font=('Times', 15)
                )
            )

        self.canvasItems = self.texts

    def toggleWordGuess(self) -> None:
        """Toggles whether AI can guess entire words"""
        self.can_guess_word = not self.can_guess_word

    def startGame(self) -> None:
        """Initializes a Hangman game"""
        playerClass = getattr(hm_players, self.selectedButton[1], None)
        self.gameGraph = None
        if playerClass is hm_players.RandomPlayer:
            self.player = playerClass()
            self.playerGraph = None
        else:
            if self.playerGraph is None:
                order = 'prev' if self.selectedButton[0] == 3 else 'next'
                graph = hm_players.load_word_bank('valid_words_large.txt', order)
                self.playerGraph = graph
            self.gameGraph = self.renderGraph(self.playerGraph)

            self.player = playerClass(self.playerGraph,
                                      can_guess_word=self.can_guess_word)

        self.hm = hangman.Hangman()
        self.hm.set_word(None)
        self.hm.set_guess_status()
        self.guessCount = 0

    def playerMakeGuess(self) -> None:
        if self.hm.game_is_finished():
            self.startGame()
            self.player._visited_characters = set()
            if self.autoPlay:
                self.d.after(400, self.playerMakeGuess)
            return

        guess = self.player.make_guess(self.hm, '')
        self.hm.make_guess(guess)
        self.guessCount += 1

        wait = 1200 if self.hm.game_is_finished() else 400
        if self.autoPlay:
            self.d.after(wait, self.playerMakeGuess)

    def togglePlay(self) -> None:
        self.autoPlay = not self.autoPlay
        if self.autoPlay:
            self.d.after(400, self.playerMakeGuess)

    def renderGraph(self, graph: hm_game_graph.GameGraph) -> np.array:
        """Renders a graph with its nodes on a circle"""
        vn = graph.get_all_vertices()
        n = len(vn)
        circle = np.array(Image.open('Assets/Circle.png'), 'float32')
        node = np.array(Image.open('Assets/Node.png'), 'float32')
        rad = 82
        off = int(circle.shape[0] / 2 - rad)
        for i in range(n):
            px = sin(2 * pi * i / n) + 1
            py = cos(2 * pi * i / n) + 1
            self.blend(circle, node, (int(px * rad) + off,
                                      int(py * rad) + off), 'alpha')
        return circle

    def renderAdjGuess(self, frame: np.array, left: int, space: int) -> None:
        """Renders edges and vertices corresponding to an adjacent guess"""
        guess = self.player.adjacent_guess(self.hm, True)
        if guess is not None:
            rad = self.gameGraph.shape[0] / 2
            offx, offy = self.W//4, self.H*3//4

            choice, known, index = guess
            vn = sorted(self.playerGraph.get_all_vertices())
            n = len(vn)
            k = vn.index(known)
            c = vn.index(choice)

            self.texts.append(self.d.create_text(
                *self.circleCoords(k / n, rad, offx, offy),
                text=known.upper(), fill='#fff', font=('Times', 18))
                )
            self.texts.append(self.d.create_text(
                *self.circleCoords(c / n, rad, offx, offy),
                text=choice.upper(), fill='#ff8', font=('Times', 18))
                )

            edges = Image.new('RGBA', self.gameGraph.shape[:2])
            draw = ImageDraw.Draw(edges)

            possible = {(w, self.playerGraph.get_weight(known, w))
                        for w in self.playerGraph.get_neighbours(known)}
            alternatives = sorted(possible, key=lambda p: p[1], reverse=True)[:5]

            for letter in alternatives:
                i = vn.index(letter[0])
                if i == c or i == k:
                    continue
                self.texts.append(self.d.create_text(
                    *self.circleCoords(i / n, rad, offx, offy),
                    text=letter[0].upper(), fill='#bcc', font=('Times', 18))
                    )
                draw.line(self.circleCoords(i / n, rad - 12, rad, rad) + \
                      self.circleCoords(k / n, rad - 12, rad, rad),
                      fill=(160, 192, 192, 255), width=2)


            draw.line(self.circleCoords(k / n, rad - 12, rad, rad) + \
                      self.circleCoords(c / n, rad - 12, rad, rad),
                      fill=(255, 255, 128, 255), width=3)

            self.blend(frame, np.array(edges), (self.W//4, self.H*3//4), 'alpha')

            x = int(type(self.player) is hm_players.GraphNextPlayer)
            self.blend(frame, self.charLight,
                       (left + space * (index + x), self.H//2), 'add')

    def circleCoords(self, i: float, r: float,
                     offx: int, offy: int) -> tuple[int, int]:
        """Returns coordinates of the point at i rounds around a circle
        centered at (offx, offy) with radius r."""
        px = sin(2 * pi * i)
        py = cos(2 * pi * i)

        return (int(px * r) + offx, int(py * r) + offy)

    def runFFGames(self) -> None:
        """Runs games fast-forward without visualization"""
        self.autoPlay = False
        eff = 0
        won = 0
        guesses = 0
        start = time.perf_counter()
        lastIdle = start

        # Indicates that we are running the games so disable the button
        self.numFFGames = -self.numFFGames

        for i in range(-self.numFFGames):
            self.player.clear_visited()
            result = hangman.run_game(self.player)
            eff += result[0] * result[1]
            won += result[1]
            guesses += len(result[2]) - 1
            # Keep the UI somewhat responsive
            if time.perf_counter() - lastIdle > 0.3:
                lastIdle = time.perf_counter()
                self.statText = 'Running... {}/{}'.format(i, -self.numFFGames)
                updateTime = time.perf_counter()
                # This is not good practice, should use multiprocessing instead
                # but since the button is disabled while running, it's okay
                self.update()
                if self.window == 'Menu':
                    break
                # Account for the time spent on GUI
                start += time.perf_counter() - updateTime

        # Indicates that we are finished running so enable the button
        self.numFFGames = -self.numFFGames

        eff /= max(1, won)
        t = time.perf_counter() - start
        stat = 'Games Won: {}\nTotal Guesses: {}\nEfficiency: {}\nTime Taken: {} s'
        self.statText = stat.format(won, guesses, round(eff, 3), round(t, 2))
        self.startGame()

    def renderVisualize(self) -> None:
        """Render the visualization screen"""
        self.totFrames += 1

        self.clearCanvas()
        self.texts = []

        frame = np.array(self.temp_bg)

        self.blend(frame, self.graphic, (self.W//2-10, self.H*2//7), 'alpha')

        if self.hm.game_is_finished():
            if self.hm.get_num_tries() == 0:
                color = np.array([[(1, 0, 0, 1.)]])
            else:
                color = np.array([[(0, 1, 0, 1.)]])

            self.blend(frame, self.graphicBlur * color,
                       (self.W//2-10, self.H*2//7), 'add')

        if self.gameGraph is not None:
            self.blend(frame, self.gameGraph, (self.W//4, self.H*3//4), 'alpha')

        for i in range(len(self.buttons)):
            self.blend(frame, self.buttons[i], self.buttonPos[i].pos, 'alpha')

        incButton = np.transpose(self.symbols[2][::2,::-2], (1,0,2))
        decButton = np.transpose(self.symbols[3][::2,::-2], (1,0,2))
        self.blend(frame, incButton, self.incFFButton.pos, 'alpha')
        self.blend(frame, decButton, self.decFFButton.pos, 'alpha')

        status = self.hm.get_guess_status().upper()
        space = int(min(50, self.W*3/4 / len(status)))
        left = (self.W - (len(status) - 1) * space) // 2

        if hasattr(self.player, 'adjacent_guess'):
            self.renderAdjGuess(frame, left, space)

        for i in range(len(status)):
            self.blend(frame, self.charBox, (left + space * i, self.H//2), 'replace')

        self.blendCursor(frame)
        self.displayImage(frame)


        name = self.d.create_text(
            self.W//6 + 30, self.H//6 + 20,
            text=self.selectedButton[1], fill='#fff',
            anchor=W, font=('Times', 22)
            )
        self.texts.append(name)

        for i in range(len(status)):
            if status[i] != '?':
                letter = self.d.create_text(
                    left + space * i, self.H//2,
                    text=status[i], fill='#fff',
                    font=('Times', 27)
                    )
                self.texts.append(letter)

        gameInfo = 'Guesses Remaining: {}\nGuesses Made: {}\nEfficiency: {}\n' \
                   '\nCorrect Word:\n{}'
        gameInfo = gameInfo.format(self.hm.get_num_tries(),
                                   self.guessCount,
                                   round(self.hm.get_efficiency_score(), 3),
                                   self.hm.get_chosen_word())
        self.texts.append(self.d.create_text(
            self.W*2//3 - 35, self.H//4 + 20, text=gameInfo,
            fill='#fff', font=('Times', 14), anchor=W
            ))

        texts = ['Faster' if self.autoPlay else 'Step',
                 'Pause' if self.autoPlay else 'Play',
                 str(abs(self.numFFGames)), 'Finish']
        for i in range(len(self.buttons)):
            self.texts.append(
                self.d.create_text(
                    *self.buttonPos[i].pos,
                    text=texts[i], fill='#fff', font=('Times', 22)
                    )
                )
        self.texts.append(
            self.d.create_text(
                self.W*2//3 - 40, self.H*3//5 + 5,
                text='Run Games:', fill='#000', font=('Times', 18)
                )
            )

        self.texts.append(
            self.d.create_text(
                self.W*2//3 - 40, self.H*3//4,
                text=self.statText,
                fill='#fff', font=('Times', 14), anchor=W
                )
            )

        self.canvasItems = self.texts


    def updateCanvas(self) -> None:
        x = self.d.winfo_pointerx() - self.d.winfo_rootx()
        y = self.d.winfo_pointery() - self.d.winfo_rooty()

        # Button updating
        for i in range(len(self.buttons)):
            self.updateButton(i, x, y, self.buttonPos[i].bounds)

        # This takes the most time
        if self.window == 'Menu':
            self.renderMenu()
        elif self.window == 'Select':
            self.renderSelect()
        elif self.window == 'Visualize':
            self.renderVisualize()

        self.after(12, self.updateCanvas)


    def updateButton(self, num, x, y, bounds):
        """Highlight a button if selected"""
        if self.selected(x, y, bounds):
            self.buttons[num] = 1.4 * self.button
        else:
            self.buttons[num] = 1.0 * self.button


    def clicked(self, evt) -> None:
        """Handle click events"""
        print(evt.x, evt.y)
        if self.window == 'Menu':
            if self.selected(evt.x, evt.y, self.buttonPos[0].bounds):
                print("Button 0 pressed")
            if self.selected(evt.x, evt.y, self.buttonPos[1].bounds):
                print("Button 1 pressed")
            if self.selected(evt.x, evt.y, self.buttonPos[2].bounds):
                self.window = 'Select'
                self.loadSelectionAssets()
            if self.selected(evt.x, evt.y, self.buttonPos[3].bounds):
                self.root.destroy()

        elif self.window == 'Select':
            for pos in self.buttonPos:
                if self.selected(evt.x, evt.y, pos.bounds):
                    self.window = 'Visualize'
                    self.loadVisualizeAssets()
                    return
            if self.selected(evt.x, evt.y, self.wordButton.bounds):
                self.toggleWordGuess()

        elif self.window == 'Visualize':
            if self.selected(evt.x, evt.y, self.buttonPos[0].bounds):
                self.playerMakeGuess()
            if self.selected(evt.x, evt.y, self.buttonPos[1].bounds):
                self.togglePlay()

            if self.numFFGames > 0:
                if self.selected(evt.x, evt.y, self.incFFButton.bounds):
                    self.numFFGames *= 10
                    self.numFFGames = min(1000000, self.numFFGames)
                elif self.selected(evt.x, evt.y, self.decFFButton.bounds):
                    self.numFFGames //= 10
                    self.numFFGames = max(10, self.numFFGames)
                elif self.selected(evt.x, evt.y, self.buttonPos[2].bounds):
                    self.runFFGames()

            if self.selected(evt.x, evt.y, self.buttonPos[3].bounds):
                self.window = 'Menu'
                self.loadMenuAssets()
                self.autoPlay = False

    def displayImage(self, frame: np.array) -> None:
        """Converts frame into Tk image and displays it on the canvas
        MUTATES frame to ensure no uint8 overflow.
        """
        frame[:,:,3] = 255
        np.minimum(frame, 255, out=frame)
        i = Image.fromarray(frame.astype("uint8"))
        self.cf = ImageTk.PhotoImage(i)
        self.d.itemconfigure(self.finalRender, image=self.cf)

    def blendCursor(self, frame: np.array) -> None:
        """Blends cursor image onto frame"""
        mx = max(0, min(self.W, self.d.winfo_pointerx() - self.d.winfo_rootx()))
        my = max(0, min(self.H, self.d.winfo_pointery() - self.d.winfo_rooty()))
        self.blend(frame, self.cursor, (mx + 20, my + 20), 'alpha')

    def clearCanvas(self) -> None:
        """Deletes items in self.canvasItems from the canvas self.d"""
        for i in self.canvasItems:
            self.d.delete(i)

    def selected(self, x, y, bounds) -> bool:
        """Return if (x,y) is inside bounds
            bounds = (left, up, right, down)
        """
        return bounds[0] < x < bounds[2] and bounds[1] < y < bounds[3]

    def blend(self, dest: np.array, source: np.array,
              coords: tuple, method="alpha") -> None:
        """Blend image source onto dest, centered at coords (x, y)

        Preconditions:
            - method in {"alpha", "add", "screen", "replace"}
        """
        left = coords[0] - (source.shape[1]//2)
        right = left + source.shape[1]
        up = coords[1] - (source.shape[0]//2)
        down = up + source.shape[0]

        src_up = 0
        src_down = source.shape[0]
        src_left = 0
        src_right = source.shape[1]

        # Clip to bounds
        if up < 0:
            src_up -= up
            up = 0
        if left < 0:
            src_left -= left
            left = 0
        if down > self.H:
            src_down -= down - self.H
            down = self.H
        if right > self.W:
            src_right -= right - self.W
            right = self.W

        source = source[src_up:src_down, src_left:src_right]

        if method == 'alpha':
            alpha = np.expand_dims(source[:,:,3], -1) / 255
            np.minimum(alpha, 1, out=alpha)
            dest[up:down, left:right] *= 1 - alpha
            dest[up:down, left:right] += source * alpha

        if method == 'add':
            dest[up:down, left:right] += source

        if method == 'screen':
            dest[up:down, left:right] = 255 - (255 - dest[up:down, left:right]) \
                                        * (255 - source) / 255

        if method == 'replace':
            dest[up:down, left:right] = source


if __name__ == "__main__":
    a = Project()
    a.start()
    a.mainloop()
    print("FPS:", a.totFrames / (time.time() - a.startTime))
