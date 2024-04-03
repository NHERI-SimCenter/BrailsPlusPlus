class Filter1(Filter):
    
    def __init__(self, input_dict):
        self.input_dict = input_dict
        
    def filter(self, image):
        print('Filter1:', image)
