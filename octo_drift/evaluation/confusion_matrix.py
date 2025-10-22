"""
Incremental confusion matrix for octo-drift evaluation.
"""
from typing import Dict, List, Set
from dataclasses import dataclass
import numpy as np


@dataclass
class Metrics:
    """Performance metrics at a given time point."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    timestamp: int
    unknown_count: float
    unknown_rate: float


class ConfusionMatrix:
    """
    Incremental confusion matrix with class merging.
    
    Tracks predictions vs true labels and supports merging
    discovered novelties back to their true classes.
    
    Attributes:
        matrix: Nested dict {true_label: {predicted_label: count}}
        last_merge: Track most recent merges
    """
    
    def __init__(self):
        self.matrix: Dict[float, Dict[float, int]] = {}
        self.last_merge: Dict[float, float] = {}
    
    def add_instance(self, true_label: float, predicted_label: float) -> None:
        """
        Add a prediction to the matrix.
        
        Args:
            true_label: Ground truth label
            predicted_label: Model prediction
        """
        # Initialize labels if new
        if true_label not in self.matrix:
            self._add_class(true_label)
        if predicted_label not in self.matrix:
            self._add_class(predicted_label)
        
        # Increment count
        self.matrix[true_label][predicted_label] += 1
    
    def _add_class(self, label: float) -> None:
        """Add a new class to the matrix."""
        self.matrix[label] = {}
        
        # Initialize with all existing labels
        for other_label in self.matrix.keys():
            self.matrix[label][other_label] = 0
            if other_label != label:
                self.matrix[other_label][label] = 0
    
    def update_confusion_matrix(self, true_label: float) -> None:
        """
        Decrement unknown (-1) count for a true label.
        Used when reclassifying previously unknown examples.
        
        Args:
            true_label: True class label
        """
        if true_label in self.matrix and -1.0 in self.matrix[true_label]:
            count = self.matrix[true_label][-1.0]
            if count > 0:
                self.matrix[true_label][-1.0] = count - 1
    
    def merge_classes(self, merge_map: Dict[float, List[float]]) -> None:
        """
        Merge discovered novelty classes back to their true labels.
        
        Args:
            merge_map: Dict mapping true_label -> [novelty_labels_to_merge]
        """
        for src_label, dest_labels in merge_map.items():
            if src_label not in self.matrix:
                continue
            
            src_row = self.matrix[src_label]
            
            for dest_label in dest_labels:
                if dest_label not in self.matrix or dest_label == src_label:
                    continue
                
                dest_row = self.matrix[dest_label]
                
                # Merge row (true class predictions)
                for col_label, count in dest_row.items():
                    src_row[col_label] = src_row.get(col_label, 0) + count
                
                # Remove merged row
                del self.matrix[dest_label]
                
                # Merge column (predictions as this class)
                for row_label in list(self.matrix.keys()):
                    if dest_label in self.matrix[row_label]:
                        count = self.matrix[row_label][dest_label]
                        self.matrix[row_label][src_label] = (
                            self.matrix[row_label].get(src_label, 0) + count
                        )
                        del self.matrix[row_label][dest_label]
                
                self.last_merge[src_label] = dest_label
        
        # Recursive merge if needed
        for src_label, dest_label in list(self.last_merge.items()):
            if dest_label in self.matrix:
                self.merge_classes({src_label: [dest_label]})
    
    def get_classes_with_nonzero_count(self) -> Dict[float, List[float]]:
        """
        Find novelty classes (>100) that should be merged to true classes (<100).
        
        Returns:
            Dict mapping true_label -> [novelty_labels_with_counts]
        """
        result = {}
        
        for true_label in self.matrix.keys():
            if 0 <= true_label < 100:  # True class range
                novelty_labels = []
                
                for pred_label in self.matrix.keys():
                    if pred_label > 100:  # Novelty range
                        count = self.matrix[true_label].get(pred_label, 0)
                        if count > 0:
                            novelty_labels.append(pred_label)
                
                if novelty_labels:
                    result[true_label] = novelty_labels
        
        return result
    
    def calculate_metrics(self, timestamp: int, 
                         unknown_count: float,
                         divisor: float) -> Metrics:
        """
        Calculate performance metrics.
        
        Args:
            timestamp: Current time
            unknown_count: Number of unknown predictions
            divisor: Divisor for time normalization
            
        Returns:
            Metrics object
        """
        true_positive = 0.0
        false_positive = 0.0
        false_negative = 0.0
        total_samples = 0.0
        
        for true_label, predictions in self.matrix.items():
            for pred_label, count in predictions.items():
                total_samples += count
                
                if true_label == pred_label:
                    true_positive += count
                else:
                    false_positive += count
                    false_negative += count
        
        true_negative = total_samples - true_positive - false_positive - false_negative
        
        # Calculate metrics
        accuracy = (true_positive + true_negative) / total_samples if total_samples > 0 else 0.0
        
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0 else 0.0
        )
        
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0 else 0.0
        )
        
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        
        unknown_rate = unknown_count / divisor if divisor > 0 else 0.0
        
        return Metrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            timestamp=timestamp,
            unknown_count=unknown_count,
            unknown_rate=unknown_rate
        )
    
    def count_unknown(self) -> int:
        """Count total unknown (-1) predictions."""
        count = 0
        for predictions in self.matrix.values():
            count += predictions.get(-1.0, 0)
        return count
    
    def get_number_of_classes(self) -> int:
        """Get total number of classes in matrix."""
        return len(self.matrix)
    
    def print_matrix(self) -> None:
        """Print confusion matrix to console."""
        print("\nConfusion Matrix:")
        print("\t", end="")
        
        labels = sorted(self.matrix.keys())
        for label in labels:
            print(f"{label}\t", end="")
        print()
        
        for true_label in labels:
            print(f"{true_label}\t", end="")
            for pred_label in labels:
                count = self.matrix[true_label].get(pred_label, 0)
                print(f"{count}\t", end="")
            print()
