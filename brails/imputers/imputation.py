from abc import ABC, abstractmethod

class Imputation(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def imputate(self, Footprint):
        pass
