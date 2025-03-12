import argparse
from datasets import load_from_disk
import random

def validate_dataset(dataset_path):
    """Validate a formatted dataset."""
    try:
        dataset = load_from_disk(dataset_path)
        
        # Check if dataset has train and validation splits
        assert "train" in dataset, "Dataset missing 'train' split"
        assert "validation" in dataset, "Dataset missing 'validation' split"
        
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
        print(f"Validation examples: {len(dataset['validation'])}")
        
        # Check test split if it exists
        if "test" in dataset:
            test_size = len(dataset["test"])
            if test_size > 0:
                test_idx = random.randint(0, test_size - 1)
                test_example = dataset["test"][test_idx]
                assert "translation" in test_example, "Test examples missing 'translation' field"
                assert "source" in test_example["translation"], "Test translation missing 'source' field"
                assert "target" in test_example["translation"], "Test translation missing 'target' field"
                print(f"Test examples: {test_size}")
                print("\nSample test example:")
                print(f"Source (kana): {test_example['translation']['source']}")
                print(f"Target (kanji): {test_example['translation']['target']}")
            else:
                print("Test split exists but contains no examples.")
        
        print("\nSample train example:")
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