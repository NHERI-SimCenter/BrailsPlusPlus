from abc import ABC, abstractmethod

class Filter(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def filter(self, image):
        pass
