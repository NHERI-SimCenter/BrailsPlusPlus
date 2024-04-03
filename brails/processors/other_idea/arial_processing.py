from abc import ABC, abstractmethod

class ArialProcessing(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def predict(self, image):
        pass
