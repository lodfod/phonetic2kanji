from transformers import BertJapaneseTokenizer
import torch

def get_japanese_tokenizer(tokenizer_type="mecab"):
    """
    Get a Japanese-specific tokenizer.
    
    Args:
        tokenizer_type (str): Type of tokenizer to use. Options: "mecab", "character"
    
    Returns:
        BertJapaneseTokenizer: A tokenizer that can handle Japanese text
    """
    if tokenizer_type == "mecab":
        # MeCab tokenizer with WordPiece
        tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    elif tokenizer_type == "character":
        # Character tokenizer
        tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
    
    return tokenizer 