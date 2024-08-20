from abc import ABC, abstractmethod


class StreetImageHarvester(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def getImage(self, Address):
        pass
