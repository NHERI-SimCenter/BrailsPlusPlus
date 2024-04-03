
# Written Kevin & Fei 03/24

from brails.processors.image_processor import ImageProcessor
from brails.types.image_set import ImageSet
from typing import Optional, Dict

class RoofShapeLLM(ImageProcessor):

    """
    The RoofShapeLLM classifier attempts to predict roof shapes using large language models.

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
        
        self.input_dict = input_dict
        
    def predict(self, image: ImageSet):

        """
        The method predicts the roof shape.
        
        Args
            images: ImageSet The set of images for which a prediction is required

        Returns
            dict: The keys being the same keys used in ImageSet.images, the values being the predicted roof shape
        """
        
        print('RoofShapeLLM: NOT YET IMPLEMENTD')
        return {}
        
