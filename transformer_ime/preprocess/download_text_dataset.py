import argparse
import os
import json
from datasets import load_dataset
from tqdm import tqdm

def download_text_dataset(output_path, dataset_name="wikipedia", subset="20220301.ja", max_samples=None):
    """
    Download a text-only Japanese dataset and save to a JSON file.
    
    Args:
        output_path: Path to save the extracted data
        dataset_name: Name of the dataset on Hugging Face
        subset: Subset of the dataset
        max_samples: Maximum number of samples to extract
    """
    print(f"Downloading {dataset_name} dataset (subset: {subset})...")
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, subset, streaming=True)
        
        # Extract text
        data = []
        for i, example in enumerate(tqdm(dataset, desc="Extracting text")):
            if max_samples and i >= max_samples:
                break
                
            if "text" in example and example["text"].strip():
                # Split into sentences
                sentences = example["text"].split("。")
                for sentence in sentences:
                    if len(sentence.strip()) > 10:  # Only keep sentences of reasonable length
                        data.append({"ja": sentence.strip() + "。"})
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Extracted {len(data)} sentences.")
        print(f"Data saved to {output_path}")
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("Falling back to sample data...")
        create_sample_dataset(output_path, 1000)

def create_sample_dataset(output_path, num_samples=1000):
    """Create a sample dataset as fallback."""
    # (Same implementation as in create_sample_dataset.py)
    # ...

def main():
    parser = argparse.ArgumentParser(description="Download a text-only Japanese dataset")
    parser.add_argument("--output", required=True, help="Path to save the extracted data")
    parser.add_argument("--dataset", default="wikipedia", help="Dataset name on Hugging Face")
    parser.add_argument("--subset", default="20220301.ja", help="Dataset subset")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to extract")
    
    args = parser.parse_args()
    download_text_dataset(args.output, args.dataset, args.subset, args.max_samples)

if __name__ == "__main__":
    main() 