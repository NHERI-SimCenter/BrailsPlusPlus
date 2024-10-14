
# Written Brian & Fei 06/10

from brails.processors.gpt_base_model.GPT import GPT
from brails.types.image_set import ImageSet
from typing import Optional, Dict

class RoofShapeGPT(GPT):

    """
    The NFLoorGPT classifier attempts to predict number of floors using GPT-4o.

    Variables
    
    Methods:
       predict(ImageSet): To return the predictions for the set of images provided

    """
    
    def __init__(self, api_key, input_dict: Optional[dict] =None):
        
        """
        The class constructor sets up the path prompts or whatever.
        
        Args
            input_data: dict Optional. The init function looks into dict for values needed, e.g. path to prompts
        """
        super().__init__(api_key, input_dict = input_dict)
        self.input_dict = input_dict
        if(self.input_dict!=None):
            self.prompt_str = self.input_dict['prompt_str']
            self.classes = self.input_dict['classes']
        else:
            self.prompt_str = "Given a satellite image containing a building roof, tell me the roof type of among hip, gable, and flat. Please choose one and the only one option and explain the reasons. Please answer the roof type by starting with 'The roof type of the building in the image is '[xxx]' where '[xxx]' is the predicted roof type. "
            self.classes = ['hip', 'gable', 'flat']
        
