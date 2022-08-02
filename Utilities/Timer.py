from time import time

class Timers:
    def __init__(self, timerNames):
        self.timers = {name:Timer(name) for name in timerNames}

    def start(self, name):
        self.timers[name].start()

    def pause(self, name):
        self.timers[name].pause()

    def pauseAll(self):
        for key in self.timers:
            self.timers[key].pause()

    def print(self):
        for key in self.timers:
            print(key, self.timers[key].totalTime)


class Timer:
    def __init__(self, name):
        self.name = name
        self.startTime = None
        self.totalTime = 0

    def start(self):
        self.startTime = time()

    def pause(self):
        if self.startTime:
            self.totalTime += time() - self.startTime
            self.startTime = None
