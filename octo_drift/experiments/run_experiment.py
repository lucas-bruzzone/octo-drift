"""
Main experiment runner for octo-drift.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List
import json

from octo_drift.core.structures import Example
from octo_drift.config.parameters import OctoDriftConfig, ExperimentConfig
from octo_drift.pipeline.offline_phase import OfflinePhase
from octo_drift.pipeline.online_phase import OnlinePhase
from octo_drift.utils.io import (
    load_arff,
    load_csv,
    save_results,
    save_metrics,
    save_novelties,
)
from octo_drift.utils.visualization import plot_metrics, plot_all_metrics
from octo_drift.evaluation.confusion_matrix import Metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_single_experiment(
    train_examples: List[Example],
    test_examples: List[Example],
    config: OctoDriftConfig,
    output_dir: Path,
) -> Metrics:
    """
    Run single experiment with given configuration.

    Args:
        train_examples: Training data
        test_examples: Test stream
        config: Algorithm configuration
        output_dir: Output directory

    Returns:
        Final metrics
    """
    logger.info("Starting offline phase...")
    offline = OfflinePhase(
        k=config.k,
        fuzzification=config.fuzzification,
        alpha=config.alpha,
        theta=config.theta,
        min_weight=config.min_weight_offline,
    )

    supervised_model = offline.train(train_examples)
    logger.info(f"Trained {len(supervised_model.get_spfmics())} micro-clusters")

    logger.info("Starting online phase...")
    online = OnlinePhase(
        supervised_model=supervised_model,
        k_short=config.k_short,
        phi=config.phi,
        latency=config.latency,
        chunk_size=config.chunk_size,
        buffer_threshold=config.buffer_threshold,
        time_threshold=config.time_threshold,
        min_weight=config.min_weight_online,
        percent_labeled=config.percent_labeled,
    )

    metrics_list, novelty_flags = online.process_stream(
        test_examples, evaluation_interval=config.evaluation_interval
    )

    logger.info(f"Processed {len(test_examples)} examples")
    logger.info(f"Final accuracy: {metrics_list[-1].accuracy:.4f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(test_examples, output_dir / "results.csv")
    save_metrics(metrics_list, output_dir / "metrics.csv")
    save_novelties(novelty_flags, output_dir / "novelties.csv")

    # Generate plots
    plot_metrics(metrics_list, novelty_flags, save_path=output_dir / "metrics_plot.png")
    plot_all_metrics(metrics_list, save_path=output_dir / "all_metrics.png")

    return metrics_list[-1]


def run_grid_search(
    train_examples: List[Example],
    test_examples: List[Example],
    exp_config: ExperimentConfig,
    output_dir: Path,
) -> Dict:
    """
    Run grid search over parameters.

    Args:
        train_examples: Training data
        test_examples: Test stream
        exp_config: Experiment configuration
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    results = {}

    for phi in exp_config.phi_values:
        for T in exp_config.buffer_threshold_values:
            for k_short in exp_config.k_short_values:
                config = exp_config.base_config
                config.phi = phi
                config.buffer_threshold = T
                config.k_short = k_short

                exp_name = f"phi{phi}_T{T}_k{k_short}"
                logger.info(f"\n{'='*50}")
                logger.info(f"Running: {exp_name}")
                logger.info(f"{'='*50}")

                exp_output = output_dir / exp_name
                final_metrics = run_single_experiment(
                    train_examples, test_examples, config, exp_output
                )

                results[exp_name] = {
                    "phi": phi,
                    "T": T,
                    "k_short": k_short,
                    "accuracy": final_metrics.accuracy,
                    "precision": final_metrics.precision,
                    "recall": final_metrics.recall,
                    "f1_score": final_metrics.f1_score,
                }

    # Save grid search results
    with open(output_dir / "grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run octo-drift experiments")
    parser.add_argument("--train", type=str, required=True, help="Training data path")
    parser.add_argument("--test", type=str, required=True, help="Test data path")
    parser.add_argument(
        "--output", type=str, default="./output", help="Output directory"
    )
    parser.add_argument("--config", type=str, help="Config JSON file")
    parser.add_argument("--grid-search", action="store_true", help="Run grid search")

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading training data: {args.train}")
    if args.train.endswith(".arff"):
        train_examples, _ = load_arff(args.train)
    else:
        train_examples = load_csv(args.train, has_header=True)

    logger.info(f"Loading test data: {args.test}")
    if args.test.endswith(".arff"):
        test_examples, _ = load_arff(args.test)
    else:
        test_examples = load_csv(args.test, has_header=True)

    logger.info(f"Train: {len(train_examples)}, Test: {len(test_examples)}")

    output_dir = Path(args.output)

    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = OctoDriftConfig.from_dict(config_dict)
    else:
        config = OctoDriftConfig()

    # Run experiment
    if args.grid_search:
        exp_config = ExperimentConfig(
            dataset_name="experiment",
            dataset_path=args.test,
            phi_values=[0.1, 0.2, 0.3],
            buffer_threshold_values=[30, 40, 50],
            k_short_values=[3, 4, 5],
            base_config=config,
        )
        run_grid_search(train_examples, test_examples, exp_config, output_dir)
    else:
        run_single_experiment(train_examples, test_examples, config, output_dir)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
