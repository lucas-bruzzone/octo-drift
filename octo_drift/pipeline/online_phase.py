"""
Online phase for octo-drift - Stream processing and novelty detection.
"""
from typing import List, Tuple
import numpy as np
from ..core.structures import Example
from ..models.supervised import SupervisedModel
from ..models.unsupervised import UnsupervisedModel
from ..detection.novelty_detector import NoveltyDetector
from ..evaluation.confusion_matrix import ConfusionMatrix, Metrics


class OnlinePhase:
    """
    Online processing phase.
    
    Processes data stream, classifies examples, detects novelties,
    and updates models incrementally with latency.
    
    Attributes:
        supervised_model: MCC (known classes)
        unsupervised_model: MCD (discovered patterns)
        novelty_detector: Novelty detection logic
        k_short: Clusters for unknown buffer
        latency: Label delay
        chunk_size: Batch size for incremental learning
        buffer_threshold: Unknown buffer size trigger
        time_threshold: Removal threshold for old clusters
        percent_labeled: Fraction of delayed labels to use
        confusion_matrix: Evaluation matrix
    """
    
    def __init__(self, supervised_model: SupervisedModel,
                 k_short: int, phi: float, latency: int,
                 chunk_size: int, buffer_threshold: int,
                 time_threshold: int, min_weight: int,
                 percent_labeled: float):
        self.supervised_model = supervised_model
        self.unsupervised_model = UnsupervisedModel()
        
        self.novelty_detector = NoveltyDetector(
            k_short=k_short,
            phi=phi,
            min_weight=min_weight,
            alpha=supervised_model.alpha,
            theta=supervised_model.theta,
            fuzzification=supervised_model.fuzzification
        )
        
        self.k_short = k_short
        self.latency = latency
        self.chunk_size = chunk_size
        self.buffer_threshold = buffer_threshold
        self.time_threshold = time_threshold
        self.percent_labeled = percent_labeled
        
        self.confusion_matrix = ConfusionMatrix()
    
    def process_stream(self, stream: List[Example],
                      evaluation_interval: int = 1000) -> Tuple[List[Metrics], List[float]]:
        """
        Process data stream with latency and novelty detection.
        
        Args:
            stream: List of examples in arrival order
            evaluation_interval: Compute metrics every N examples
            
        Returns:
            Tuple of (metrics_list, novelty_flags)
        """
        unknown_buffer = []
        labeled_buffer = []
        delayed_stream = []
        
        metrics_list = []
        novelty_flags = []
        
        latency_counter = 0
        delayed_idx = 0
        
        for timestamp, example in enumerate(stream):
            # Classify with MCC
            predicted_label = self.supervised_model.classify(example, timestamp)
            example.predicted_label = predicted_label
            
            # If unknown, try MCD
            if predicted_label == -1:
                predicted_label = self.unsupervised_model.classify(
                    example, 
                    self.supervised_model.k, 
                    timestamp
                )
                example.predicted_label = predicted_label
                
                # Still unknown - add to buffer
                if predicted_label == -1:
                    unknown_buffer.append(example)
                    
                    # Trigger novelty detection
                    if len(unknown_buffer) >= self.buffer_threshold:
                        unknown_buffer, novelty_detected = self.novelty_detector.detect(
                            unknown_buffer,
                            self.supervised_model,
                            self.unsupervised_model,
                            timestamp
                        )
                        
                        if novelty_detected:
                            # Update confusion matrix with discovered patterns
                            for ex in unknown_buffer:
                                if ex.predicted_label != -1:
                                    self.confusion_matrix.update_confusion_matrix(
                                        ex.true_label
                                    )
            
            # Update confusion matrix
            self.confusion_matrix.add_instance(
                example.true_label,
                example.predicted_label
            )
            
            # Store for delayed learning
            delayed_stream.append(example)
            latency_counter += 1
            
            # Process delayed labels
            if latency_counter >= self.latency:
                if delayed_idx < len(delayed_stream):
                    delayed_example = delayed_stream[delayed_idx]
                    
                    # Probabilistic labeling
                    if np.random.random() < self.percent_labeled or not labeled_buffer:
                        labeled_buffer.append(delayed_example)
                    
                    # Incremental training
                    if len(labeled_buffer) >= self.chunk_size:
                        labeled_buffer = self.supervised_model.train_new_classifier(
                            labeled_buffer,
                            timestamp
                        )
                        labeled_buffer.clear()
                    
                    delayed_idx += 1
            
            # Cleanup old clusters
            self.supervised_model.remove_old_spfmics(
                self.latency + self.time_threshold,
                timestamp
            )
            self.unsupervised_model.remove_old_spfmics(
                self.latency + self.time_threshold,
                timestamp
            )
            self._remove_old_unknown(unknown_buffer, self.time_threshold, timestamp)
            
            # Periodic evaluation
            if timestamp > 0 and timestamp % evaluation_interval == 0:
                # Merge discovered novelties to true classes
                merge_map = self.confusion_matrix.get_classes_with_nonzero_count()
                self.confusion_matrix.merge_classes(merge_map)
                
                # Calculate metrics
                unknown_count = self.confusion_matrix.count_unknown()
                metrics = self.confusion_matrix.calculate_metrics(
                    timestamp,
                    unknown_count,
                    evaluation_interval
                )
                metrics_list.append(metrics)
                
                # Track novelties
                # Check if any novelty was detected in this interval
                novelty_in_interval = any(
                    ex.predicted_label >= 100 
                    for ex in stream[max(0, timestamp - evaluation_interval):timestamp]
                )
                novelty_flags.append(1.0 if novelty_in_interval else 0.0)
        
        return metrics_list, novelty_flags
    
    def _remove_old_unknown(self, buffer: List[Example], 
                           threshold: int, 
                           current_time: int) -> None:
        """Remove old examples from unknown buffer."""
        buffer[:] = [
            ex for ex in buffer
            if current_time - ex.timestamp < threshold
        ]
