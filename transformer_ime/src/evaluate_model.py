import argparse
import torch
import os
from transformers import AutoModelForSeq2SeqLM
from japanese_tokenizer import get_japanese_tokenizer
import evaluate
from tqdm import tqdm

def load_data(data_dir):
    """Load kana and kanji data from the specified directory."""
    kana_path = os.path.join(data_dir, "data.kana")
    kanji_path = os.path.join(data_dir, "data.kanji")
    
    if not os.path.exists(kana_path) or not os.path.exists(kanji_path):
        raise FileNotFoundError(f"Could not find data files in {data_dir}")
    
    with open(kana_path, 'r', encoding='utf-8') as f:
        kana_lines = [line.strip() for line in f if line.strip()]
    
    with open(kanji_path, 'r', encoding='utf-8') as f:
        kanji_lines = [line.strip() for line in f if line.strip()]
    
    if len(kana_lines) != len(kanji_lines):
        raise ValueError(f"Mismatch in number of lines: {len(kana_lines)} kana lines vs {len(kanji_lines)} kanji lines")
    
    print(f"Loaded {len(kana_lines)} examples from {data_dir}")
    return kana_lines, kanji_lines

def main():
    parser = argparse.ArgumentParser(description="Evaluate a kana to kanji model using CER")
    parser.add_argument("--model_dir", required=True, help="Directory with the fine-tuned model")
    parser.add_argument("--data_dir", default="data/reazon", help="Directory containing data.kana and data.kanji files")
    parser.add_argument("--tokenizer_type", default="mecab", choices=["mecab", "character"], 
                        help="Type of Japanese tokenizer to use")
    parser.add_argument("--output_file", default="model_evaluation.txt", 
                        help="File to save the evaluation results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to evaluate (None for all)")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_dir}...")
    
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    
    # Load a Japanese-specific tokenizer
    tokenizer = get_japanese_tokenizer(args.tokenizer_type)
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using {device} device")
    
    model = model.to(device)
    
    # Load data
    kana_lines, kanji_lines = load_data(args.data_dir)
    
    # Limit samples if specified
    if args.max_samples is not None:
        kana_lines = kana_lines[:args.max_samples]
        kanji_lines = kanji_lines[:args.max_samples]
    
    # Load CER metric
    cer_metric = evaluate.load("cer")
    
    # Open output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write("Source (Kana)\tPrediction\tReference (Kanji)\n")
        
        all_predictions = []
        all_references = kanji_lines
        
        # Process in batches
        for i in range(0, len(kana_lines), args.batch_size):
            batch_kana = kana_lines[i:i+args.batch_size]
            batch_kanji = kanji_lines[i:i+args.batch_size]
            
            # Prepare inputs
            inputs = ["translate kana to kanji: " + kana for kana in batch_kana]
            input_encodings = tokenizer(inputs, padding=True, truncation=True, 
                                       return_tensors="pt").to(device)
            
            # Generate predictions
            with torch.no_grad():
                try:
                    # Get the maximum input length in this batch
                    max_input_length = input_encodings["input_ids"].shape[1]
                    # Set max_length to be proportional to input length
                    dynamic_max_length = min(max_input_length * 2, 128)
                    
                    output_ids = model.generate(
                        input_ids=input_encodings["input_ids"],
                        attention_mask=input_encodings["attention_mask"],
                        max_length=dynamic_max_length,
                        num_beams=4,
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False
                    )
                except RuntimeError as e:
                    print(f"Error during generation: {e}")
                    print("Falling back to greedy decoding...")
                    # Fallback to simpler generation method
                    output_ids = model.generate(
                        input_ids=input_encodings["input_ids"],
                        attention_mask=input_encodings["attention_mask"],
                        max_length=128,
                        num_beams=1,  # No beam search
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )
            
            # Decode predictions
            predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            all_predictions.extend(predictions)
            
            # Write results to file
            for kana, pred, kanji in zip(batch_kana, predictions, batch_kanji):
                f.write(f"{kana}\t{pred}\t{kanji}\n")
            
            # Calculate and print batch CER
            batch_cer = cer_metric.compute(predictions=predictions, references=batch_kanji)
            print(f"Batch {i//args.batch_size + 1} CER: {batch_cer}")
        
        # Calculate final CER for all examples
        final_cer = cer_metric.compute(predictions=all_predictions, references=all_references)
        print(f"Final CER: {final_cer}")
        f.write(f"\nOverall CER: {final_cer}")
        
        # Calculate some additional statistics
        correct_count = sum(1 for pred, ref in zip(all_predictions, all_references) if pred == ref)
        accuracy = correct_count / len(all_predictions) if all_predictions else 0
        print(f"Exact match accuracy: {accuracy:.4f} ({correct_count}/{len(all_predictions)})")
        f.write(f"\nExact match accuracy: {accuracy:.4f} ({correct_count}/{len(all_predictions)})")

if __name__ == "__main__":
    main()
