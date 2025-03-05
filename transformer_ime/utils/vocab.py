import argparse
from pathlib import Path
import json
from tqdm import tqdm

def extract_unique_chars(file_path):
    """Extract all unique characters from a text file."""
    unique_chars = set()
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing lines"):
            # Add each character to the set
            unique_chars.update(line.strip())
    
    # Convert to sorted list for consistent ordering
    char_list = sorted(list(unique_chars))
    
    # Create vocabulary dictionary with indices
    vocab_dict = {char: idx for idx, char in enumerate(char_list)}
    
    # Add special tokens
    vocab_dict["[PAD]"] = len(vocab_dict)
    vocab_dict["[UNK]"] = len(vocab_dict)
    
    return vocab_dict

def main():
    parser = argparse.ArgumentParser(description="Create vocabulary from text file")
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--output", default="vocab.json", help="Path to output vocabulary file")
    
    args = parser.parse_args()
    
    # Extract vocabulary
    vocab = extract_unique_chars(args.input)
    
    # Print statistics
    print(f"\nTotal unique characters: {len(vocab) - 2}")  # subtract 2 for special tokens
    print("\nSample characters:")
    for char, idx in list(vocab.items())[:10]:
        print(f"{char}: {idx}")
    
    # Save vocabulary
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"\nVocabulary saved to {output_path}")

if __name__ == "__main__":
    main()