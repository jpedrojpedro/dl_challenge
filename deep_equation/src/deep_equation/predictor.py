"""
Predictor interfaces for the Deep Learning challenge.
"""
import torch
import numpy as np
from typing import List
from pathlib import Path
from deep_equation.src.deep_equation.model import LeNet
from deep_equation.src.deep_equation.helper import adjust_inputs


class BaseNet:
    """
    Base class that must be used as base interface to implement 
    the predictor using the model trained by the student.
    """

    def load_model(self, model_path):
        """
        Implement a method to load models given a model path.
        """
        pass

    def predict(
        self, 
        images_a: List, 
        images_b: List, 
        operators: List[str], 
        device: str = 'cpu'
    ) -> List[float]:
        """
        Make a batch prediction considering a mathematical operator 
        using digits from image_a and image_b.
        Instances from iamges_a, images_b, and operators are aligned:
            - images_a[0], images_b[0], operators[0] -> regards the 0-th input instance
        Args: 
            * images_a (List[PIL.Image]): List of RGB PIL Image of any size
            * images_b (List[PIL.Image]): List of RGB PIL Image of any size
            * operators (List[str]): List of mathematical operators from ['+', '-', '*', '/']
                - invalid options must return `None`
            * device: 'cpu' or 'cuda'
        Return: 
            * predicted_number (List[float]): the list of numbers representing the result of the equation from the inputs: 
                [{digit from image_a} {operator} {digit from image_b}]
        """
    # do your magic

    pass 


class RandomModel(BaseNet):
    """This is a dummy random classifier, it is not using the inputs
        it is just an example of the expected inputs and outputs
    """

    def load_model(self, model_path):
        """
        Method responsible for loading the model.
        If you need to download the model, 
        you can download and load it inside this method.
        """
        np.random.seed(42)

    def predict(
        self, images_a, images_b,
        operators, device='cpu'
    ) -> List[float]:

        predictions = []
        for _image_a, _image_b, _operator in zip(images_a, images_b, operators):
            random_prediction = np.random.uniform(-10, 100, size=1)[0]
            predictions.append(random_prediction)
        return predictions


class StudentModel(BaseNet):
    """
    TODO: THIS is the class you have to implement:
        load_model: method that loads your best model.
        predict: method that makes batch predictions.
    """

    def load_model(self, model_path="model_state/20211103-173938_state.pt"):
        base_path = Path(__file__)
        model_full_path = base_path.parent.parent.parent / model_path
        model = LeNet()
        model.load_state_dict(torch.load(model_full_path))
        model.eval()
        return model

    # TODO:
    def predict(
        self, images_a, images_b,
        operators, device='cpu'
    ):
        """Implement this method to perform predictions 
        given a list of images_a, images_b and operators.
        """
        model = self.load_model()
        fixed_inputs = adjust_inputs(images_a, images_b, operators)
        predictions = model(fixed_inputs)
        predictions = [float(torch.argmax(pred)) for pred in predictions]
        return predictions
