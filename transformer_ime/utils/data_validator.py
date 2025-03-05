import argparse
from datasets import load_from_disk
import random

def validate_dataset(dataset_path):
    """Validate a formatted dataset."""
    try:
        dataset = load_from_disk(dataset_path)
        
        # Check if dataset has train and test splits
        assert "train" in dataset, "Dataset missing 'train' split"
        assert "test" in dataset, "Dataset missing 'test' split"
        
        # Check if examples have the expected format
        train_size = len(dataset["train"])
        random_idx = random.randint(0, train_size - 1)
        example = dataset["train"][random_idx]
        assert "translation" in example, "Examples missing 'translation' field"
        assert "source" in example["translation"], "Translation missing 'source' field"
        assert "target" in example["translation"], "Translation missing 'target' field"
        
        # Print dataset statistics
        print(f"Dataset validation successful!")
        print(f"Train examples: {len(dataset['train'])}")
        print(f"Test examples: {len(dataset['test'])}")
        print("\nSample example:")
        print(f"Source (kana): {example['translation']['source']}")
        print(f"Target (kanji): {example['translation']['target']}")
        
        return True
    except Exception as e:
        print(f"Dataset validation failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate a formatted dataset")
    parser.add_argument("--dataset_path", required=True, help="Path to the formatted dataset directory")
    
    args = parser.parse_args()
    validate_dataset(args.dataset_path)

if __name__ == "__main__":
    main() 