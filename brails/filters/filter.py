from abc import ABC, abstractmethod
from brails.types.image_set import ImageSet

class Filter(ABC):
    def __init__(self, input_data: dict):
        self.input_data = input_data
    
    @abstractmethod
    def filter(self, images_in: ImageSet, images_out: ImageSet):
        pass
