from brails.processing.arial_processing import ArialProcessing

class RoofShape(ArialProcessing):
    
    def __init__(self, input_dict):
        self.input_dict = input_dict
        
    def predict(self, image):
        print('RoofShape:', image)
