class Filter2(Filter):
    
    def __init__(self, input_dict):
        self.input_dict = input_dict
        
    def filter(self, image):
        print('Filter2:', image)
