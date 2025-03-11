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

def create_hf_dataset(kana_lines, kanji_lines, train_ratio=0.8, test_ratio=0.1, validation_ratio=0.1):
    """
    Create a Hugging Face dataset with train, test, and validation splits.
    
    Args:
        kana_lines: List of kana sentences
        kanji_lines: List of kanji sentences
        train_ratio: Ratio of data to use for training (default: 0.8)
        test_ratio: Ratio of data to use for testing (default: 0.1)
        validation_ratio: Ratio of data to use for validation (default: 0.1)
        
    Returns:
        DatasetDict with 'train', 'test', and 'validation' splits
    """
    # Verify that ratios sum to 1.0
    total_ratio = train_ratio + test_ratio + validation_ratio
    if not 0.999 <= total_ratio <= 1.001:  # Allow for small floating-point errors
        raise ValueError(f"Train, test, and validation ratios must sum to 1.0, got {total_ratio}")
    
    data = {
        "translation": [
            {"source": kana, "target": kanji} 
            for kana, kanji in zip(kana_lines, kanji_lines)
        ]
    }
    
    # Create dataset directly with the translation format
    dataset = Dataset.from_dict(data)
    
    # First split into train and temp (test + validation)
    train_temp = dataset.train_test_split(train_size=train_ratio, seed=42)
    
    # Calculate the ratio of test to (test + validation)
    remaining_ratio = test_ratio + validation_ratio
    test_ratio_of_temp = test_ratio / remaining_ratio if remaining_ratio > 0 else 0
    
    # Split the temp into test and validation
    test_validation = train_temp['test'].train_test_split(
        train_size=test_ratio_of_temp, 
        seed=42
    )
    
    # Create the final dataset dictionary
    final_dataset = DatasetDict({
        'train': train_temp['train'],
        'test': test_validation['train'],
        'validation': test_validation['test']
    })
    
    # Print split statistics
    print(f"Dataset split statistics:")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Train split: {len(final_dataset['train'])} examples ({len(final_dataset['train'])/len(dataset):.1%})")
    print(f"  Test split: {len(final_dataset['test'])} examples ({len(final_dataset['test'])/len(dataset):.1%})")
    print(f"  Validation split: {len(final_dataset['validation'])} examples ({len(final_dataset['validation'])/len(dataset):.1%})")
    
    return final_dataset

def main():
    parser = argparse.ArgumentParser(description="Format kana-kanji data for Hugging Face transformers")
    parser.add_argument("--kana_file", required=True, help="Path to the kana file")
    parser.add_argument("--kanji_file", required=True, help="Path to the kanji file")
    parser.add_argument("--output_dir", required=True, help="Directory to save the formatted dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training (default: 0.8)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of data to use for testing (default: 0.1)")
    parser.add_argument("--validation_ratio", type=float, default=0.1, help="Ratio of data to use for validation (default: 0.1)")
    
    args = parser.parse_args()
    
    # Verify that ratios sum to 1.0
    total_ratio = args.train_ratio + args.test_ratio + args.validation_ratio
    if not 0.999 <= total_ratio <= 1.001:  # Allow for small floating-point errors
        parser.error(f"Train, test, and validation ratios must sum to 1.0, got {total_ratio}")
    
    # Load data
    kana_lines, kanji_lines = load_kana_kanji_files(args.kana_file, args.kanji_file)
    
    # Create dataset with direct translation format and 3-way split
    dataset = create_hf_dataset(
        kana_lines, 
        kanji_lines, 
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio
    )
    
    # Save dataset
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    print(f"Dataset saved to {args.output_dir}")

if __name__ == "__main__":
    main() 