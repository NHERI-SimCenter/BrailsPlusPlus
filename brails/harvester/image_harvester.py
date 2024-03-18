from abc import ABC, abstractmethod

class ImageHarvester(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def getImage(self, Address):
        pass
