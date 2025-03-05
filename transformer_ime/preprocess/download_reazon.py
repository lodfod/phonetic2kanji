import argparse
import os
import json
from datasets import load_dataset
from tqdm import tqdm
import traceback

def download_reazon_dataset(output_path, size="tiny", split=None, max_samples=None):
    """
    Download the ReazonSpeech dataset and save the transcriptions to a JSON file.
    
    Args:
        output_path: Path to save the extracted data
        size: Dataset size to use ('tiny', 'small', 'medium', 'large', 'all')
        split: Dataset split to use (None, 'train', 'validation', 'test')
        max_samples: Maximum number of samples to extract
    """
    print(f"Downloading ReazonSpeech dataset (size: {size}, split: {split or 'all'})...")
    
    try:
        print("Loading dataset...")
        dataset = load_dataset("reazon-research/reazonspeech", size, trust_remote_code=True)
        
        print("Dataset loaded. Structure:", dataset)
        
        # Determine which split to use
        if split is not None and split in dataset:
            working_dataset = dataset[split]
            print(f"Using split: {split}")
        else:
            # If no specific split is requested or the requested split doesn't exist use the first available split
            available_splits = list(dataset.keys())
            if available_splits:
                split = available_splits[0]
                working_dataset = dataset[split]
                print(f"No specific split requested or requested split not found. Using split: {split}")
            else:
                raise ValueError("Dataset has no splits available")
        
        print(f"Working dataset has {len(working_dataset)} examples")
        
        # Extract transcriptions
        transcriptions = []
        count = 0
        
        # Limit the number of examples if max_samples is specified
        examples_to_process = working_dataset
        if max_samples is not None and max_samples < len(working_dataset):
            examples_to_process = working_dataset.select(range(max_samples))
            print(f"Limited to {max_samples} examples")
        
        for i, example in enumerate(tqdm(examples_to_process, desc="Extracting transcriptions")):
            if max_samples is not None and i >= max_samples:
                break
            try:
                if "transcription" in example and example["transcription"].strip():
                    transcriptions.append({"transcription": example["transcription"].strip()})
                    count += 1
                    
            except Exception as e:
                print(f"Error processing example {i}: {str(e)}")
                continue
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=2)
        
        print(f"Extracted {len(transcriptions)} transcriptions.")
        print(f"Data saved to {output_path}")
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        

def main():
    parser = argparse.ArgumentParser(description="Download and extract ReazonSpeech dataset")
    parser.add_argument("--output", required=True, help="Path to save the extracted data")
    parser.add_argument("--size", default="tiny", choices=["tiny", "small", "medium", "large", "all"], 
                        help="Dataset size to use")
    parser.add_argument("--split", default=None, choices=[None, "train", "validation", "test"], 
                        help="Dataset split to use (if applicable)")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to extract")
    
    args = parser.parse_args()
    download_reazon_dataset(args.output, args.size, args.split, args.max_samples)

if __name__ == "__main__":
    main() 