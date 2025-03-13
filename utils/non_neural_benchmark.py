import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re
import unicodedata
import Levenshtein

def normalize_text(text):
    """Normalize text by removing spaces and normalizing unicode characters"""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', '', text)  # Remove spaces
    return text

def calculate_metrics(reference, prediction):
    """
    Calculate metrics for a single pair of reference and prediction
    
    Args:
        reference (str): Reference text (ground truth)
        prediction (str): Predicted text
        
    Returns:
        dict: Dictionary containing precision, recall, CER, sentence accuracy, and F-score
    """
    # Normalize texts
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)
    
    # Character-level evaluation
    ref_chars = set(reference)
    pred_chars = set(prediction)
    
    # Calculate true positives, false positives, false negatives
    true_positives = sum(1 for c in pred_chars if c in ref_chars)
    false_positives = sum(1 for c in pred_chars if c not in ref_chars)
    false_negatives = sum(1 for c in ref_chars if c not in pred_chars)
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F-score
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate Character Error Rate (CER)
    distance = Levenshtein.distance(reference, prediction)
    cer = distance / len(reference) if len(reference) > 0 else 1.0
    
    # Calculate sentence accuracy (1 if exact match, 0 otherwise)
    sentence_accuracy = 1.0 if reference == prediction else 0.0
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f_score': f_score * 100,
        'cer': cer * 100,
        'sentence_accuracy': sentence_accuracy * 100
    }

def evaluate_files(prediction_file, reference_file, debug=False):
    """
    Evaluate prediction file against reference file
    
    Args:
        prediction_file (Path): Path to prediction file
        reference_file (Path): Path to reference file
        debug (bool): Whether to print debug information
        
    Returns:
        dict: Dictionary containing average metrics
    """
    # Read files
    with open(prediction_file, 'r', encoding='utf-8') as f:
        pred_lines = f.read().splitlines()
    
    with open(reference_file, 'r', encoding='utf-8') as f:
        ref_lines = f.read().splitlines()
    
    if len(pred_lines) != len(ref_lines):
        raise ValueError(f"Number of lines in prediction file ({len(pred_lines)}) and reference file ({len(ref_lines)}) don't match")
    
    # Initialize metrics
    all_metrics = []
    
    # Process each line
    for pred, ref in tqdm(zip(pred_lines, ref_lines), total=len(pred_lines), desc="Evaluating"):
        # Calculate metrics
        metrics = calculate_metrics(ref, pred)
        all_metrics.append(metrics)
        
        if debug:
            print(f"Reference: {ref}")
            print(f"Prediction: {pred}")
            print(f"Metrics: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F={metrics['f_score']:.2f}, CER={metrics['cer']:.2f}, Acc={metrics['sentence_accuracy']}")
            print("-" * 80)
    
    # Calculate average metrics
    avg_metrics = {
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f_score': np.mean([m['f_score'] for m in all_metrics]),
        'cer': np.mean([m['cer'] for m in all_metrics]),
        'sentence_accuracy': np.mean([m['sentence_accuracy'] for m in all_metrics])
    }
    
    return avg_metrics

def main():
    """Main function to parse arguments and evaluate files"""
    parser = argparse.ArgumentParser(description='Evaluate baseline kana-to-kanji conversion')
    parser.add_argument('--prediction', type=str, required=True, help='Path to prediction .kanji file')
    parser.add_argument('--reference', type=str, required=True, help='Path to reference .kanji file')
    parser.add_argument('--debug', action='store_true', help='Print individual results for debugging')
    
    args = parser.parse_args()
    
    # Convert paths
    prediction_file = Path(args.prediction)
    reference_file = Path(args.reference)
    
    # Evaluate files
    print(f"Evaluating {prediction_file} against {reference_file}...")
    avg_metrics = evaluate_files(prediction_file, reference_file, args.debug)
    
    # Print overall results
    print("\n\nOverall Results:")
    print("=" * 80)
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"F-score: {avg_metrics['f_score']:.4f}")
    print(f"Character Error Rate (CER): {avg_metrics['cer']:.4f}")
    print(f"Sentence Accuracy: {avg_metrics['sentence_accuracy']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()