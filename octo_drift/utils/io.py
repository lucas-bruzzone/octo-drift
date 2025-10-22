"""
File I/O utilities for octo-drift.
"""
from typing import List, Tuple
import csv
import numpy as np
from pathlib import Path
from scipy.io import arff
import pandas as pd
from ..core.structures import Example
from ..evaluation.confusion_matrix import Metrics


def load_arff(filepath: str) -> Tuple[List[Example], List[str]]:
    """
    Load ARFF file and convert to Examples.
    
    Args:
        filepath: Path to ARFF file
        
    Returns:
        Tuple of (examples_list, attribute_names)
    """
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    
    # Convert bytes to string if needed
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    
    examples = []
    for idx, row in df.iterrows():
        example = Example.from_array(
            row.values.astype(np.float64),
            has_label=True,
            timestamp=idx
        )
        examples.append(example)
    
    return examples, list(df.columns)


def load_csv(filepath: str, has_header: bool = False) -> List[Example]:
    """
    Load CSV file and convert to Examples.
    
    Args:
        filepath: Path to CSV file
        has_header: Whether first row is header
        
    Returns:
        List of Examples
    """
    examples = []
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        if has_header:
            next(reader)
        
        for idx, row in enumerate(reader):
            data = np.array([float(x) for x in row])
            example = Example.from_array(data, has_label=True, timestamp=idx)
            examples.append(example)
    
    return examples


def save_results(examples: List[Example], filepath: str) -> None:
    """
    Save classification results to CSV.
    
    Args:
        examples: List of classified examples
        filepath: Output path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'True_Label', 'Predicted_Label'])
        
        for idx, ex in enumerate(examples, start=1):
            pred_label = 'unknown' if ex.predicted_label == -1 else ex.predicted_label
            writer.writerow([idx, ex.true_label, pred_label])


def save_metrics(metrics_list: List[Metrics], filepath: str) -> None:
    """
    Save metrics to CSV.
    
    Args:
        metrics_list: List of Metrics objects
        filepath: Output path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Timestamp', 'Accuracy', 'Precision', 'Recall', 
            'F1_Score', 'Unknown_Count', 'Unknown_Rate'
        ])
        
        for m in metrics_list:
            writer.writerow([
                m.timestamp,
                m.accuracy,
                m.precision,
                m.recall,
                m.f1_score,
                m.unknown_count,
                m.unknown_rate
            ])


def save_novelties(novelty_flags: List[float], filepath: str) -> None:
    """
    Save novelty detection flags to CSV.
    
    Args:
        novelty_flags: Binary flags (1=novelty detected, 0=none)
        filepath: Output path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Novelty'])
        
        for idx, flag in enumerate(novelty_flags):
            writer.writerow([idx, flag])


def load_metrics(filepath: str) -> List[Metrics]:
    """
    Load metrics from CSV.
    
    Args:
        filepath: Path to metrics CSV
        
    Returns:
        List of Metrics objects
    """
    metrics_list = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics = Metrics(
                accuracy=float(row['Accuracy']),
                precision=float(row['Precision']),
                recall=float(row['Recall']),
                f1_score=float(row['F1_Score']),
                timestamp=int(row['Timestamp']),
                unknown_count=float(row['Unknown_Count']),
                unknown_rate=float(row['Unknown_Rate'])
            )
            metrics_list.append(metrics)
    
    return metrics_list


def load_novelties(filepath: str) -> List[float]:
    """
    Load novelty flags from CSV.
    
    Args:
        filepath: Path to novelties CSV
        
    Returns:
        List of novelty flags
    """
    flags = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            flags.append(float(row['Novelty']))
    
    return flags
