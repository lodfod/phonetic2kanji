from transformers import BertJapaneseTokenizer
from datasets import load_dataset
import os

def train_japanese_tokenizer(data_files, output_dir, tokenizer_type="mecab", vocab_size=32000):
    """
    Train a Japanese tokenizer on your data.
    
    Args:
        data_files: Dictionary mapping split names to file paths
        output_dir: Directory to save the tokenizer
        tokenizer_type: Type of Japanese tokenizer to use as base
        vocab_size: Size of the vocabulary
    """
    # Load the dataset
    dataset = load_dataset('text', data_files=data_files)
    
    # Initialize with BERT Japanese tokenizer
    if tokenizer_type == "mecab":
        base_tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    elif tokenizer_type == "character":
        base_tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
    
    # Define tokenizer training arguments
    tokenizer_args = {
        "vocab_size": vocab_size,
        "min_frequency": 2,
        "special_tokens": ["<pad>", "<unk>", "<s>", "</s>"],
    }
    
    # Train the tokenizer
    def batch_iterator():
        for i in range(0, len(dataset["train"]), 1000):
            yield dataset["train"][i:i+1000]["text"]
    
    new_tokenizer = base_tokenizer.train_new_from_iterator(batch_iterator(), **tokenizer_args)
    
    # Save the tokenizer
    os.makedirs(output_dir, exist_ok=True)
    new_tokenizer.save_pretrained(output_dir)
    
    return new_tokenizer

if __name__ == "__main__":
    # Define data files
    data_files = {
        "train": "transformer_ime/data/reazon/data.kana",
        # Add more files if needed
    }
    
    # Train and save the tokenizer
    tokenizer = train_japanese_tokenizer(data_files, "models/tokenizer")
    print(f"Tokenizer saved to models/tokenizer")
    
    # Test the tokenizer with Japanese text
    test_text = "コトハ"
    tokens = tokenizer.tokenize(test_text)
    print(f"Test tokenization of '{test_text}': {tokens}") 