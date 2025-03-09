from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import re
import unicodedata
import Levenshtein
import torch

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

def evaluate_model(kana_file, kanji_file, model, tokenizer, device, max_length, debug):
    """
    Evaluate model on kana-kanji pairs using direct model access
    
    Args:
        kana_file (Path): Path to kana file
        kanji_file (Path): Path to kanji file
        model: The model to use for translation
        tokenizer: The tokenizer to use
        device: The device to run on
        max_length: Maximum length for generation
        debug: Whether to print debug information
        
    Returns:
        dict: Dictionary containing average metrics
    """
    # Read files
    with open(kana_file, 'r', encoding='utf-8') as f:
        kana_lines = f.read().splitlines()
    
    with open(kanji_file, 'r', encoding='utf-8') as f:
        kanji_lines = f.read().splitlines()
    
    if len(kana_lines) != len(kanji_lines):
        raise ValueError(f"Number of lines in kana file ({len(kana_lines)}) and kanji file ({len(kanji_lines)}) don't match")
    
    # Initialize metrics
    all_metrics = []
    
    # Process each line
    for kana, kanji_reference in tqdm(zip(kana_lines, kanji_lines), total=len(kana_lines), desc="Evaluating"):
        # Format input with prefix
        input_text = f"translate kana to kanji: {kana}"
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Generate
        outputs = model.generate(inputs.input_ids, max_length=max_length)
        
        # Decode
        kanji_prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate metrics
        metrics = calculate_metrics(kanji_reference, kanji_prediction)
        all_metrics.append(metrics)
        
        if debug:
            print(f"Input (kana): {kana}")
            print(f"Formatted input: {input_text}")
            print(f"Reference (kanji): {kanji_reference}")
            print(f"Prediction (kanji): {kanji_prediction}")
            print(f"Metrics: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F={metrics['f_score']:.4f}, CER={metrics['cer']:.4f}, Acc={metrics['sentence_accuracy']}")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate kana-to-kanji translation model')
    parser.add_argument('--kana', type=str, required=True, help='Path to kana file')
    parser.add_argument('--kanji', type=str, required=True, help='Path to kanji file')
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--max_length', type=int, default=128, help='Max length of model generation')
    parser.add_argument('--debug', action='store_true', help='Print individual results for debugging')
    
    args = parser.parse_args()
    
    # Convert paths
    kana_file = Path(args.kana)
    kanji_file = Path(args.kanji)
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using {device} device")
    model = model.to(device)
    
    # Evaluate model
    avg_metrics = evaluate_model(kana_file, kanji_file, model, tokenizer, device, args.max_length, args.debug)
    
    # Print overall results
    print("\n" + "=" * 80)
    print("Overall Results:")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"F-score: {avg_metrics['f_score']:.4f}")
    print(f"Character Error Rate (CER): {avg_metrics['cer']:.4f}")
    print(f"Sentence Accuracy: {avg_metrics['sentence_accuracy']:.4f}")
    print("=" * 80)