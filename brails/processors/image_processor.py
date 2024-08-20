
from abc import ABC, abstractmethod
from brails.types.image_set import ImageSet

class ImageProcessor(ABC):
    
    """
    The abstract interface defining the methods an ImageProcessing class must implement.

    Methods:
       predict(ImageSet): To return the predictions for the set of images provided

    """    
    # OTHER POTENTIAL ABSTRATCT METHODS:
    #    train(TrainingImageSet): ->model
    #    retrain(TrainingImageSet, model): ->model
    #  ??? does predict get passed the model .. if so what is Model classes API

    
    @abstractmethod
    def predict(self, images: ImageSet) ->dict:
        """
        The pure virtual method that will be called to make the prediction

        Args:
            images (ImageSet): The set of images for which predictions are required
        
        Returns:
            dict: The keys being the same keys used in ImageSet.images, the values being a single prediction of an array of predictions
        """        
        pass
