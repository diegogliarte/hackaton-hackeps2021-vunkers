import cv2
from pynput import keyboard


class Manager:

    def __init__(self, config):
        self.config = config
        self.realtime = config["realtime"]
        self.gaussian = config["effects"]["gaussian"]
        self.threshold = config["effects"]["threshold"]
        self.debug = config["debug"]
        self.min_yellow = 0
        self.max_yellow = 255

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

    def on_press(self, key):
        try:
            key = key.char.upper()
            if key == "Q":
                self.gaussian = max(1, self.gaussian - 2)
            elif key == "W":
                self.gaussian += 2
            elif key == "O":
                self.threshold = max(0, self.threshold - 1)
            elif key == "P":
                self.threshold += 1
            elif key == "R":
                self.realtime = not self.realtime
            elif key == "D":
                self.debug = not self.debug
                cv2.destroyWindow("debug")

        except:
            pass
