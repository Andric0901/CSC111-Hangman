# Main menu

from tkinter import *
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import time
from math import sin


f = ('Times', 32, 'bold')

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

        # Open image assets
        bg = Image.open('Assets/Background.jpg').convert('RGBA')
        self.background = np.array(bg, 'float32')

        title = Image.open('Assets/Title.png')
        self.title = np.array(title, 'float32')

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

        names = Image.open('Assets/Names.png').convert('RGBA').resize((400, 36))
        self.names = np.array(names, 'float32')

        button = Image.open('Assets/Button.png')
        self.button = np.array(button, 'float32')

        # Copy so we can change the intensity of each button individually
        self.buttons = [np.array(self.button),
                        np.array(self.button),
                        np.array(self.button),
                        np.array(self.button)]

        cursor = Image.open('Assets/Cursor.png').resize((53, 50))
        self.cursor = np.clip(np.array(cursor, 'float32') * 1.8, None, 255)

    def start(self) -> None:
        """Start the rendering loop"""
        self.makeWidgets()
        self.render()
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


    def render(self) -> None:
        """Render the main menu"""
        self.totFrames += 1

        # Copy background
        frame = np.array(self.background)

        # Blend in title and names
        self.blend(frame, self.title, (self.W//2, self.H//8), 'add')
        self.blend(frame, self.graphic, (self.W//2, self.H//3), 'add')
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
        self.blend(frame, self.buttons[0], (self.W//4, self.H-360), 'alpha')
        self.blend(frame, self.buttons[1], (self.W//4, self.H-180), 'alpha')
        self.blend(frame, self.buttons[2], (self.W*3//4, self.H-360), 'alpha')
        self.blend(frame, self.buttons[3], (self.W*3//4, self.H-180), 'alpha')

        # Blend cursor
        mx = max(0, min(self.W, self.d.winfo_pointerx() - self.d.winfo_rootx()))
        my = max(0, min(self.H, self.d.winfo_pointery() - self.d.winfo_rooty()))
        self.blend(frame, self.cursor, (mx + 20, my + 20), 'alpha')

        # Convert numpy array to image
        frame[:,:,3] = 255
        frame = np.clip(frame, 0, 255)
        i = Image.fromarray(frame.astype("uint8"))
        self.cf = ImageTk.PhotoImage(i)
        self.d.itemconfigure(self.finalRender, image=self.cf)

        self.clearCanvas()
        
        # Add text to buttons
        self.text0 = self.d.create_text(self.W*1//4, self.H-360,
                                        text="Instructions", fill="#fff", font=f)

        self.text1 = self.d.create_text(self.W*1//4, self.H-180,
                                        text="Play!", fill="#fff", font=f)

        self.text2 = self.d.create_text(self.W*3//4, self.H-360,
                                        text="Compare AI", fill="#fff", font=f)

        self.text3 = self.d.create_text(self.W*3//4, self.H-180,
                                        text="Quit", fill="#fff", font=f)

        self.canvasItems = [self.text0, self.text1, self.text2, self.text3]


    def updateCanvas(self) -> None:
        x = self.d.winfo_pointerx() - self.d.winfo_rootx()
        y = self.d.winfo_pointery() - self.d.winfo_rooty()

        # Button updating
        self.updateButton(0, x, y, (100, 200, 380, 280))
        self.updateButton(1, x, y, (100, 380, 380, 460))
        self.updateButton(2, x, y, (580, 200, 860, 280))
        self.updateButton(3, x, y, (580, 380, 860, 460))

        # This takes the most time
        self.render()

        if self.window == 'Menu':
            self.after(12, self.updateCanvas)


    def updateButton(self, num, x, y, bounds):
        """Highlight a button if selected"""
        if self.selected(x, y, bounds):
            self.buttons[num] = 1.3 * self.button
        else:
            self.buttons[num] = 1.0 * self.button


    def clicked(self, evt) -> None:
        """Handle click events"""
        if self.window == 'Menu':
            if self.selected(evt.x, evt.y, (100, 240, 380, 320)):
                print("Button 0 pressed")
            if self.selected(evt.x, evt.y, (100, 380, 380, 460)):
                print("Button 1 pressed")
            if self.selected(evt.x, evt.y, (500, 240, 780, 320)):
                print("Button 2 pressed")

            if self.selected(evt.x, evt.y, (500, 380, 780, 460)):
                print("Quit")
                self.root.destroy()


        elif self.window == 'Graph':
            if self.selected(evt.x, evt.y, (100, 480, 380, 550)):
                if self.country == 'C':
                    self.country = 'U'
                else:
                    self.country = 'C'

            if self.selected(evt.x, evt.y, (500, 480, 780, 550)):
                self.window = "Menu"

                self.clearCanvas()
                self.updateCanvas()

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
            - method in {"alpha", "add", "screen"}
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
            dest[up:down, left:right] *= 1 - alpha
            dest[up:down, left:right] += source * alpha

        if method == 'add':
            dest[up:down, left:right] += source

        if method == 'screen':
            dest[up:down, left:right] = 255 - (255 - dest[up:down, left:right]) \
                                        * (255 - source) / 255


if __name__ == "__main__":
    a = Project()
    a.start()
    a.mainloop()
    print("FPS:", a.totFrames / (time.time() - a.startTime))
