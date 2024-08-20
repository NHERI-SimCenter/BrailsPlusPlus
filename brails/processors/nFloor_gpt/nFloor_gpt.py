
# Written Brian & Fei 06/10

from brails.processors.gpt_base_model.GPT import GPT
from brails.types.image_set import ImageSet
from typing import Optional, Dict

class NFloorGPT(GPT):

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
            input_data: dict Optional. The init function looks into dict for values needed, e.g. path to promts
        """
        super().__init__(api_key, input_dict = input_dict)
        self.input_dict = input_dict
        if(self.input_dict!=None):
            self.prompt_str = self.input_dict['prompt_str']
            self.classes = self.input_dict['classes']
        else:
            self.prompt_str = "Given a streetview image of a house, tell me the number of stories of the house, where a story is defined as any level part of a building that has a floor and a ceiling and is occupied or intended for occupancy. If the uppermost floor look like 0.5 stories, please trim/round based on its window relative to other windows, as well as the feasibility of space for occupancy. Please answer the task by starting with 'The number of stories of the house in the image is '[xxx]' where '[xxx]' is (1)one-story (2)two-story (3)three-story. "
            self.classes = ['one-story', 'two-story', 'three-story']

        
