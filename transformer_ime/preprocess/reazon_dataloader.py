import argparse
import os
import json
import random
from tqdm import tqdm
import MeCab
import jaconv
import unicodedata
import string

def format_text(text):
    """Format text by normalizing and removing punctuation."""
    text = unicodedata.normalize("NFKC", text)  
    table = str.maketrans("", "", string.punctuation + "「」、。・")
    text = text.translate(table)
    return text

def load_reazon_data(data_path):
    """Load data from Reazon dataset JSON file."""
    print(f"Loading data from {data_path}...")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract Japanese sentences from the dataset
        sentences = []
        if isinstance(data, list):
            # If data is a list of dictionaries
            for item in tqdm(data, desc="Processing items"):
                if isinstance(item, dict) and 'transcription' in item and item['transcription'].strip():
                    sentences.append(item['transcription'].strip())
                elif isinstance(item, dict) and 'ja' in item and item['ja'].strip():
                    sentences.append(item['ja'].strip())
        elif isinstance(data, dict) and 'train' in data:
            # If data has a 'train' key (common in some dataset formats)
            for item in tqdm(data['train'], desc="Processing train items"):
                if 'transcription' in item and item['transcription'].strip():
                    sentences.append(item['transcription'].strip())
                elif 'ja' in item and item['ja'].strip():
                    sentences.append(item['ja'].strip())
        
        return sentences
    except json.JSONDecodeError:
        # If not a JSON file, try reading as text file with one sentence per line
        print("Not a valid JSON file. Trying to read as text file...")
        with open(data_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        return sentences

def main():
    parser = argparse.ArgumentParser(description="Extract Japanese text from Reazon dataset")
    parser.add_argument("--input", required=True, help="Path to Reazon dataset file")
    parser.add_argument("--output", required=True, help="Path to output file for kanji text")
    parser.add_argument("--max_sentences", type=int, default=None, help="Maximum number of sentences to extract")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the sentences")
    parser.add_argument("--mecab_path", help="Path to MeCab dictionary (optional)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load data
    sentences = load_reazon_data(args.input)
    
    if not sentences:
        print("No sentences found in the input file.")
        return
    
    # Format sentences if needed
    formatted_sentences = []
    for sentence in tqdm(sentences, desc="Formatting sentences"):
        # You can add additional formatting here if needed
        formatted_sentences.append(sentence)
    
    # Shuffle if requested
    if args.shuffle:
        print("Shuffling sentences...")
        random.shuffle(formatted_sentences)
    
    # Limit number of sentences if specified
    if args.max_sentences and len(formatted_sentences) > args.max_sentences:
        print(f"Limiting to {args.max_sentences} sentences...")
        formatted_sentences = formatted_sentences[:args.max_sentences]
    
    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(formatted_sentences))
    
    print(f"Extracted {len(formatted_sentences)} sentences.")
    print(f"Data saved to {args.output}")

if __name__ == "__main__":
    main()