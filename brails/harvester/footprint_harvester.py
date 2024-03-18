from abc import ABC, abstractmethod

class FootprintHarvester(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def getFootprints(self, Location):
        pass
