import os
from pathlib import Path
from datasets import load_dataset
import argparse
from tqdm import tqdm

def extract_transcriptions(dataset_name, output_file):
    """
    Extract transcriptions from a Japanese ASR dataset and save them to a .kanji file.
    
    Args:
        dataset_name (str): Name of the dataset to load ('jsut_basic5000' or 'common_voice_8_0')
        output_file (Path): Path to save the output .kanji file
    """
    # Load the appropriate dataset
    if dataset_name == 'jsut_basic5000':
        dataset = load_dataset("japanese-asr/ja_asr.jsut_basic5000", split="test")
    elif dataset_name == 'common_voice_8_0':
        dataset = load_dataset("japanese-asr/ja_asr.common_voice_8_0", split="test")
    else:
        raise ValueError("Dataset must be either 'jsut_basic5000' or 'common_voice_8_0'")
    
    print(f"Loaded {dataset_name} dataset with {len(dataset)} examples")
    
    # Extract transcriptions with progress bar
    transcriptions = []
    for example in tqdm(dataset, desc="Extracting transcriptions", unit="example"):
        transcriptions.append(example['transcription'])  
    
    # Create directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file with progress bar
    with output_file.open('w', encoding='utf-8') as f:
        for i, transcription in enumerate(tqdm(transcriptions, desc="Writing to file", unit="line")):
            if i < len(transcriptions) - 1:
                f.write(f"{transcription}\n")
            else:
                f.write(f"{transcription}")  # No newline for the last sentence
    
    print(f"Saved {len(transcriptions)} transcriptions to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract transcriptions from Japanese ASR datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['jsut_basic5000', 'common_voice_8_0'],
                        help='Dataset to use (jsut_basic5000 or common_voice_8_0)')
    parser.add_argument('--output', type=str, default='data/jsut_basic/data.kanji',
                        help='Output file path (default: data/jsut_basic/data.kanji)')
    
    args = parser.parse_args()
    
    # Convert output path to Path object
    output_path = Path(args.output)
    
    extract_transcriptions(args.dataset, output_path)