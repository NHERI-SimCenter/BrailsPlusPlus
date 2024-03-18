from abc import ABC, abstractmethod

class ArialHarvester(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def getImage(self, Address):
        pass
