from transformers import AutoTokenizer, T5Tokenizer
from datasets import load_from_disk

dataset = load_from_disk("data/reazon/formatted/train")

def get_training_corpus():
    for item in dataset:
        if isinstance(item, dict):
            if "text" in item:
                yield item["text"]
        elif isinstance(item, str):
            yield item

def main():
    print(dataset)
    old_tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    print(old_tokenizer)
    tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), 5000)
    tokenizer.save_pretrained("tokenizers/reazon_tokenizer")

if __name__ == "__main__":
    main()