import torch
import os
from typing import List, Union

from .agent import Agent

class Dummy(Agent):

    def __init__(
        self,
        device="cuda",
        *args,
        **kwargs,
    ):
        self.device = device
        super().__init__(wrapper=kwargs.get("wrapper", None))


    def _score(self, image_paths: Union[str, 'PIL.Image', list], question: str, target_token="Yes", candidates: list=None, temperature=0.5, *args,**kwargs):
        """
        Scores the likelihood of the target token (Yes/No) for a given question and image.
        This is a dummy method that returns a random score.
        """
        return torch.rand(1).to(self.device).squeeze(0)

    def _score_batch(self, image_paths: List[str], questions: List[str], target_token="Yes", temperature=0.5, *args,**kwargs):
        return [torch.rand(1).to(self.device).squeeze(0) for _ in range(len(image_paths))]

    def _score_with_mask(self, image: 'PIL.Image', question: str, bboxes: list, candidates: list=None, type="default", history=None, temperature=0.5,*args,**kwargs):
        """
        Scores the likelihood of the target token (Yes/No) for a given question and image.
        This is a dummy method that returns a random score.
        """
        return torch.rand(1).to(self.device).squeeze(0)
    
    def _score_contrastive(
        self, 
        image_masked_path: str, 
        image_not_masked_path: str,
        question: str, 
        target_token="Yes", 
        alpha=1.0,
        *args,
        **kwargs
    ):
        """
        Scores the likelihood of the target token (Yes/No) for a given question and pair of images.
        This is a dummy method that returns a random score.
        """
        return torch.rand(1).to(self.device).squeeze(0)

    def _query(
        self, 
        image_path: str, 
        question: str, 
        max_new_tokens=32,
        *args,
        **kwargs
    ) -> str:
        
        """
        General question-answering with the model.
        Returns the decoded text from the model's generation.

        Parameters:
        image_path: Path to the image.
        question: The question to ask.
        max_new_tokens: Maximum tokens to generate.

        Returns:
        The decoded text.
        """
        return "This is a dummy response. [[11, 121, 102, 200]]"

    def _query_batch(
        self, 
        image_paths: Union[str, 'PIL.Image', list], 
        questions: Union[str, 'PIL.Image', list], 
        max_new_tokens=128,
        batch_size=10,  # Set your desired batch size here
        *args,
        **kwargs
    ) -> Union[str, list]:
        return ["This is a dummy response. [[11, 121, 102, 200]]" for _ in range(len(image_paths))]