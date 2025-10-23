"""
Online phase for octo-drift - Stream processing and novelty detection.
"""

import logging
import time
from typing import List, Tuple
import numpy as np
from ..core.structures import Example
from ..models.supervised import SupervisedModel
from ..models.unsupervised import UnsupervisedModel
from ..detection.novelty_detector import NoveltyDetector
from ..evaluation.confusion_matrix import ConfusionMatrix, Metrics

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        supervised_model: SupervisedModel,
        k_short: int,
        phi: float,
        latency: int,
        chunk_size: int,
        buffer_threshold: int,
        time_threshold: int,
        min_weight: int,
        percent_labeled: float,
    ):
        self.supervised_model = supervised_model
        self.unsupervised_model = UnsupervisedModel()

        self.novelty_detector = NoveltyDetector(
            k_short=k_short,
            phi=phi,
            min_weight=min_weight,
            alpha=supervised_model.alpha,
            theta=supervised_model.theta,
            fuzzification=supervised_model.fuzzification,
        )

        self.k_short = k_short
        self.latency = latency
        self.chunk_size = chunk_size
        self.buffer_threshold = buffer_threshold
        self.time_threshold = time_threshold
        self.percent_labeled = percent_labeled

        self.confusion_matrix = ConfusionMatrix()

    def process_stream(
        self, stream: List[Example], evaluation_interval: int = 1000
    ) -> Tuple[List[Metrics], List[float], List[Example]]:
        """
        Process data stream with latency and novelty detection.

        Args:
            stream: List of examples in arrival order
            evaluation_interval: Compute metrics every N examples

        Returns:
            Tuple of (metrics_list, novelty_flags, processed_stream)
        """
        unknown_buffer = []
        labeled_buffer = []
        delayed_stream = []

        metrics_list = []
        novelty_flags = []

        latency_counter = 0
        delayed_idx = 0

        total_examples = len(stream)
        start_time = time.time()
        classify_time = 0
        novelty_time = 0
        train_time = 0

        logger.info(f"Processing {total_examples} examples...")

        for timestamp, example in enumerate(stream):
            # Log progress every 1000 examples
            if timestamp > 0 and timestamp % 1000 == 0:
                elapsed = time.time() - start_time
                rate = timestamp / elapsed
                eta = (total_examples - timestamp) / rate if rate > 0 else 0

                logger.info(
                    f"Progress: {timestamp}/{total_examples} "
                    f"({100*timestamp/total_examples:.1f}%) | "
                    f"Rate: {rate:.1f} ex/s | "
                    f"ETA: {eta/60:.1f}min | "
                    f"MCC: {len(self.supervised_model.get_spfmics())} | "
                    f"MCD: {len(self.unsupervised_model.get_spfmics())} | "
                    f"Buffer: {len(unknown_buffer)}"
                )
                logger.debug(
                    f"Time breakdown - Classify: {classify_time:.2f}s, "
                    f"Novelty: {novelty_time:.2f}s, Train: {train_time:.2f}s"
                )

            # Classify with MCC
            t0 = time.time()
            predicted_label = self.supervised_model.classify(example, timestamp)
            example.predicted_label = predicted_label

            # If unknown, try MCD
            if predicted_label == -1:
                predicted_label = self.unsupervised_model.classify(
                    example, self.supervised_model.k, timestamp
                )
                example.predicted_label = predicted_label

                # Still unknown - add to buffer
                if predicted_label == -1:
                    unknown_buffer.append(example)

                    # Trigger novelty detection
                    if len(unknown_buffer) >= self.buffer_threshold:
                        t1 = time.time()
                        logger.debug(f"Triggering novelty detection at t={timestamp}")

                        unknown_buffer, novelty_detected = self.novelty_detector.detect(
                            unknown_buffer,
                            self.supervised_model,
                            self.unsupervised_model,
                            timestamp,
                        )

                        novelty_time += time.time() - t1

                        if novelty_detected:
                            logger.info(f"Novelty detected at t={timestamp}")
                            # Update confusion matrix with discovered patterns
                            for ex in unknown_buffer:
                                if ex.predicted_label != -1:
                                    self.confusion_matrix.update_confusion_matrix(
                                        ex.true_label
                                    )

            classify_time += time.time() - t0

            # Update confusion matrix
            self.confusion_matrix.add_instance(
                example.true_label, example.predicted_label
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
                        t2 = time.time()
                        logger.debug(f"Incremental training at t={timestamp}")

                        labeled_buffer = self.supervised_model.train_new_classifier(
                            labeled_buffer, timestamp
                        )
                        labeled_buffer.clear()

                        train_time += time.time() - t2

                    delayed_idx += 1

            # Cleanup old clusters
            self.supervised_model.remove_old_spfmics(
                self.latency + self.time_threshold, timestamp
            )
            self.unsupervised_model.remove_old_spfmics(
                self.latency + self.time_threshold, timestamp
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
                    timestamp, unknown_count, evaluation_interval
                )
                metrics_list.append(metrics)

                logger.info(
                    f"Metrics at t={timestamp}: "
                    f"Acc={metrics.accuracy:.4f}, "
                    f"Prec={metrics.precision:.4f}, "
                    f"UnkRate={metrics.unknown_rate:.4f}"
                )

                # Track novelties
                novelty_in_interval = any(
                    ex.predicted_label >= 100
                    for ex in stream[
                        max(0, timestamp - evaluation_interval) : timestamp
                    ]
                )
                novelty_flags.append(1.0 if novelty_in_interval else 0.0)

        total_time = time.time() - start_time
        logger.info(f"Processing complete in {total_time/60:.2f} minutes")
        logger.info(f"Average rate: {total_examples/total_time:.1f} examples/second")

        return metrics_list, novelty_flags, stream

    def _remove_old_unknown(
        self, buffer: List[Example], threshold: int, current_time: int
    ) -> None:
        """Remove old examples from unknown buffer."""
        buffer[:] = [ex for ex in buffer if current_time - ex.timestamp < threshold]
