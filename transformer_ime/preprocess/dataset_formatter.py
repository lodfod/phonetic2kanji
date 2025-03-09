import os
import argparse
from datasets import Dataset, DatasetDict

def load_kana_kanji_files(kana_file, kanji_file):
    """Load kana and kanji files and return them as lists of sentences."""
    with open(kana_file, 'r', encoding='utf-8') as f_kana:
        kana_lines = [line.strip() for line in f_kana if line.strip()]
    
    with open(kanji_file, 'r', encoding='utf-8') as f_kanji:
        kanji_lines = [line.strip() for line in f_kanji if line.strip()]
    
    assert len(kana_lines) == len(kanji_lines), "Number of lines in kana and kanji files don't match"
    return kana_lines, kanji_lines

def create_hf_dataset(kana_lines, kanji_lines, train_ratio=0.8):
    """Create a Hugging Face dataset directly with source and target."""
    data = {
        "translation": [
            {"source": kana, "target": kanji} 
            for kana, kanji in zip(kana_lines, kanji_lines)
        ]
    }
    
    # Create dataset directly with the translation format
    dataset = Dataset.from_dict(data)
    
    # Split into train and test
    train_test_split = dataset.train_test_split(train_size=train_ratio, seed=42)
    
    return train_test_split

def main():
    parser = argparse.ArgumentParser(description="Format kana-kanji data for Hugging Face transformers")
    parser.add_argument("--kana_file", required=True, help="Path to the kana file")
    parser.add_argument("--kanji_file", required=True, help="Path to the kanji file")
    parser.add_argument("--output_dir", required=True, help="Directory to save the formatted dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training")
    
    args = parser.parse_args()
    
    # Load data
    kana_lines, kanji_lines = load_kana_kanji_files(args.kana_file, args.kanji_file)
    
    # Create dataset with direct translation format
    dataset = create_hf_dataset(kana_lines, kanji_lines, args.train_ratio)
    
    # Save dataset
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    print(f"Dataset saved to {args.output_dir}")

if __name__ == "__main__":
    main() 