import threading
import time
import random

class ActivePool(object):
    def __init__(self):
        super(ActivePool, self).__init__()
        self.active = []
        self.lock = threading.Lock()


    def makeActive(self, name):
        with self.lock:
            self.active.append(name)


    def makeInactive(self, name):
        with self.lock:
            self.active.remove(name)


    def numActive(self):
        with self.lock:
            return len(self.active)
            
            
    def __str__(self):
        with self.lock:
            return str(self.active)