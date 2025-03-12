import argparse
from datasets import load_from_disk

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract test data to kana and kanji files.")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--kana_output', type=str, required=True, help='Path to save the test kana file.')
    parser.add_argument('--kanji_output', type=str, required=True, help='Path to save the test kanji file.')
    
    args = parser.parse_args()

    # Load the dataset from disk
    dataset = load_from_disk(args.dataset_path)

    # Extract the test dataset
    test_dataset = dataset['test']

    # Open files to write the kana and kanji data
    with open(args.kana_output, 'w', encoding='utf-8') as kana_file, open(args.kanji_output, 'w', encoding='utf-8') as kanji_file:
        for i, example in enumerate(test_dataset):
            # Extract the source (kana) and target (kanji) from each example
            kana = example['translation']['source']
            kanji = example['translation']['target']
            
            # Write to the respective files
            kana_file.write(kana)
            kanji_file.write(kanji)
            
            # Add a newline if it's not the last line
            if i < len(test_dataset) - 1:
                kana_file.write('\n')
                kanji_file.write('\n')

    print(f"Files '{args.kana_output}' and '{args.kanji_output}' have been created.")

if __name__ == "__main__":
    main()