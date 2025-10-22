"""
Supervised model (MCC - Known Classes Model) for octo-drift.
"""
from typing import List, Dict
import numpy as np
from ..core.structures import Example, SPFMiC
from ..core.distance import euclidean_distance, calculate_pertinence
from ..core.fuzzy_functions import (
    FuzzyCMeans, 
    separate_by_class, 
    create_spfmics_from_clusters
)


class SupervisedModel:
    """
    Model of Known Classes (MCC).
    
    Maintains fuzzy micro-clusters for each known class and performs
    classification based on typicality and pertinence.
    
    Attributes:
        k: Number of clusters per class
        fuzzification: Fuzzification parameter
        alpha: Pertinence exponent
        theta: Typicality exponent
        min_weight: Minimum cluster size
        known_labels: Set of known class labels
        classifier: Dict mapping label -> list of SPFMiCs
    """
    
    def __init__(self, k: int, fuzzification: float, alpha: float, 
                 theta: float, min_weight: int):
        self.k = k
        self.fuzzification = fuzzification
        self.alpha = alpha
        self.theta = theta
        self.min_weight = min_weight
        self.known_labels: List[float] = []
        self.classifier: Dict[float, List[SPFMiC]] = {}
    
    def train_initial_model(self, examples: List[Example]) -> None:
        """
        Train initial model from labeled data.
        
        Args:
            examples: List of labeled examples
        """
        # Separate by class
        examples_by_class = separate_by_class(examples)
        
        # Train micro-clusters for each class
        for label, class_examples in examples_by_class.items():
            if len(class_examples) > self.k:
                if label not in self.known_labels:
                    self.known_labels.append(label)
                
                # Cluster and create SPFMiCs
                clusterer = FuzzyCMeans(self.k, self.fuzzification)
                clusterer.fit(class_examples)
                
                spfmics = create_spfmics_from_clusters(
                    class_examples, 
                    clusterer, 
                    label,
                    self.alpha, 
                    self.theta, 
                    self.min_weight, 
                    timestamp=0
                )
                
                self.classifier[label] = spfmics
    
    def classify(self, example: Example, timestamp: int) -> float:
        """
        Classify example using MCC.
        
        Args:
            example: Example to classify
            timestamp: Current time
            
        Returns:
            Predicted label or -1 if outlier
        """
        all_spfmics = self.get_spfmics()
        
        typicalities = []
        pertinences = []
        candidate_spfmics = []
        
        # Find micro-clusters within radius
        for spfmic in all_spfmics:
            distance = euclidean_distance(example, spfmic)
            
            if distance <= spfmic.radius_weighted:
                typicality = spfmic.calculate_typicality(
                    example.point, 
                    self.k, 
                    distance
                )
                pertinence = calculate_pertinence(
                    example.point, 
                    spfmic.centroid, 
                    self.fuzzification
                )
                
                typicalities.append(typicality)
                pertinences.append(pertinence)
                candidate_spfmics.append(spfmic)
        
        # No matching cluster - outlier
        if not candidate_spfmics:
            return -1.0
        
        # Select cluster with max typicality
        max_idx = np.argmax(typicalities)
        selected_spfmic = candidate_spfmics[max_idx]
        max_pertinence = pertinences[max_idx]
        
        # Update micro-cluster
        distance = euclidean_distance(example, selected_spfmic)
        selected_spfmic.assign_example(example, max_pertinence, 1.0, distance)
        selected_spfmic.updated = timestamp
        
        return selected_spfmic.label
    
    def train_new_classifier(self, examples: List[Example], 
                            timestamp: int) -> List[Example]:
        """
        Incrementally train new micro-clusters from labeled batch.
        
        Args:
            examples: Labeled examples
            timestamp: Current time
            
        Returns:
            Examples that couldn't form valid clusters
        """
        remaining = []
        examples_by_class = separate_by_class(examples)
        
        for label, class_examples in examples_by_class.items():
            if len(class_examples) >= self.k * 2:
                if label not in self.known_labels:
                    self.known_labels.append(label)
                
                clusterer = FuzzyCMeans(self.k, self.fuzzification)
                clusterer.fit(class_examples)
                
                spfmics = create_spfmics_from_clusters(
                    class_examples,
                    clusterer,
                    label,
                    self.alpha,
                    self.theta,
                    self.min_weight,
                    timestamp
                )
                
                # Add or merge with existing
                if label not in self.classifier:
                    self.classifier[label] = spfmics
                else:
                    self.classifier[label].extend(spfmics)
            else:
                remaining.extend(class_examples)
        
        return remaining
    
    def get_spfmics(self) -> List[SPFMiC]:
        """Get all SPFMiCs from all classes."""
        all_spfmics = []
        for spfmic_list in self.classifier.values():
            all_spfmics.extend(spfmic_list)
        return all_spfmics
    
    def remove_old_spfmics(self, threshold: int, current_time: int) -> None:
        """
        Remove obsolete micro-clusters.
        
        Args:
            threshold: Time window threshold
            current_time: Current timestamp
        """
        for label in list(self.classifier.keys()):
            spfmics = self.classifier[label]
            
            # Keep only recent micro-clusters
            self.classifier[label] = [
                spfmic for spfmic in spfmics
                if not (current_time - spfmic.created > threshold and 
                       current_time - spfmic.updated > threshold)
            ]
            
            # Remove class if no micro-clusters remain
            if not self.classifier[label]:
                del self.classifier[label]
