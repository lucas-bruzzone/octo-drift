"""
Offline phase for octo-drift - Initial model training.
"""
from typing import List
from ..core.structures import Example
from ..models.supervised import SupervisedModel


class OfflinePhase:
    """
    Offline training phase.
    
    Trains initial MCC from labeled training data using
    fuzzy c-means clustering per class.
    
    Attributes:
        k: Number of clusters per class
        fuzzification: Fuzzification parameter
        alpha: Pertinence exponent
        theta: Typicality exponent
        min_weight: Minimum cluster size
    """
    
    def __init__(self, k: int, fuzzification: float, alpha: float,
                 theta: float, min_weight: int):
        self.k = k
        self.fuzzification = fuzzification
        self.alpha = alpha
        self.theta = theta
        self.min_weight = min_weight
    
    def train(self, training_examples: List[Example]) -> SupervisedModel:
        """
        Train initial supervised model.
        
        Args:
            training_examples: Labeled training data
            
        Returns:
            Trained SupervisedModel (MCC)
        """
        model = SupervisedModel(
            k=self.k,
            fuzzification=self.fuzzification,
            alpha=self.alpha,
            theta=self.theta,
            min_weight=self.min_weight
        )
        
        model.train_initial_model(training_examples)
        
        return model
