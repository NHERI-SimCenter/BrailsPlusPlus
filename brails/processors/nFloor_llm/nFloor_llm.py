
# Written Brian & Fei 03/24

from brails.processors.vlm_image_classifier.CLIPClassifier import CLIPClassifier
from brails.types.image_set import ImageSet
from typing import Optional, Dict

class NFloorLLM(CLIPClassifier):

    """
    The NFLoorLLM classifier attempts to predict number of floors using large language models.

    Variables
    
    Methods:
       predict(ImageSet): To return the predictions for the set of images provided

    """
    
    def __init__(self, input_dict: Optional[dict] =None):
        
        """
        The class constructor sets up the path prompts or whatever.
        
        Args
            input_data: dict Optional. The init function looks into dict for values needed, e.g. path to promts
        """
        super().__init__(task = "roofshape", input_dict = input_dict)
        self.input_dict = input_dict
        if(self.input_dict!=None):
            self.text_prompts = self.args['prompts']
            self.classes = self.args['classes']
        else:
            #each class should have equal amount of text prompts
            self.text_prompts = [
                'one story house', 'bungalow', 'flat house', 'single-story side split house',
                'two story house', 'two story townhouse', 'side split house', 'raised ranch', 
                'three story house','three story house', 'three story house', 'three-decker'
            ]
            self.classes = [1, 2, 3]
        
    # inherit method from CLIPClassifier
    # def predict(self, image: ImageSet):

    #     """
    #     The method predicts the roof shape.
        
    #     Args
    #         images: ImageSet The set of images for which a prediction is required

    #     Returns
    #         dict: The keys being the same keys used in ImageSet.images, the values being the predicted roof shape
    #     """
        
        
    #     return
        
