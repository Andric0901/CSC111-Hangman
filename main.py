# CSC 111 Final Project
# Main script
#
# Running this file will first extract image assets
# from the zip folder and then start a GUI window.

import zipfile
import time
from Visualizer import Project

if __name__ == '__main__':
    zf = zipfile.ZipFile('Assets.zip')
    zf.extractall()

    gui = Project()
    gui.start()
    gui.mainloop()
    print('Average FPS:', gui.totFrames / (time.time() - gui.startTime))
