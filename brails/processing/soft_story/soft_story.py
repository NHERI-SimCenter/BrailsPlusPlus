from brails.processing.street_processor import StreetProcessor

class SoftStory(StreetProcessor):
    
    def __init__(self, input_dict):
        self.input_dict = input_dict
        
    def predict(self, image):
        print('SoftStory:', image)
