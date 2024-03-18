from brails.processing.street_processor import StreetProcessor

class FacadeParser(StreetProcessor):
    
    def __init__(self, input_dict):
        self.input_dict = input_dict
        
    def predict(self, image):
        print('FacadeParser:', image)
