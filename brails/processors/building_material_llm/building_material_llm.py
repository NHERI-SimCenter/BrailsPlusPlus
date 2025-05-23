
# Written Brian & Fei 03/24

from brails.processors.vlm_image_classifier.CLIPClassifier import CLIPClassifier
from brails.types.image_set import ImageSet
from typing import Optional, Dict

class BuildingMaterialLLM(CLIPClassifier):

    """
    The BuildingMaterialLLM classifier attempts to predict building materials  using large language models.

    Variables
    
    Methods:
       predict(ImageSet): To return the predictions for the set of images provided

    """
    
    def __init__(self, input_dict: Optional[dict] =None):
        
        """
        The class constructor sets up the path prompts or whatever.
        
        Args
            input_data: dict Optional. The init function looks into dict for values needed, e.g. path to prompts
        """
        super().__init__(task = "roofshape", input_dict = input_dict)
        self.input_dict = input_dict
        if(self.input_dict!=None):
            self.text_prompts = self.args['prompts']
            self.classes = self.args['classes']
        else:
            self.text_prompts = ['wood', 'steel', 'concrete', 'manufactured', 'masonry']
            self.classes = ['wood', 'steel', 'concrete', 'manufactured', 'masonry']
        self.template = "a photo of a house made of {}" 
        
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
        
