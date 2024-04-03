
# Barbaros 04/24

from brails.processors.image_processor import ImageProcessor
from brails.types.image_set import ImageSet
from typing import Optional, Dict

class FacadeParser(ImageProcessor):

    """
    The FacadeParser attempts to predict a whole unch of stuff

    Variables
    
    Methods:
       predict(ImageSet): To return the predictions for the set of images provided

    """
    
    def __init__(self, input_dict: Optional[dict] =None):
        
        """
        The class constructor sets up the path prompts or whatever.
        
        Args
            input_data: dict Optional. The init function looks into dict for values needed,
        """
        
        self.input_dict = input_dict
        
    def predict(self, image: ImageSet):

        """
        The method predicts the stuff 
        
        Args
            images: ImageSet The set of images for which a prediction is required

        Return
            dict: The keys being the same keys used in ImageSet.images, the values being a list containing the predictions
        """
        
        print('FacadeParser: NOT YET IMPLEMENTD')
        return {}
