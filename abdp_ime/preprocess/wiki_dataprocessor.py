import os
import glob
import json
import MeCab
import jaconv
import unicodedata
import string
from tqdm import tqdm
import argparse
from pathlib import Path

def format_text(text):
    """Format given text by normalizing and removing punctuation."""
    text = unicodedata.normalize("NFKC", text)  
    table = str.maketrans("", "", string.punctuation + "「」、。・")
    text = text.translate(table)
    return text

def add_spaces(text, mecab_tagger):
    """Add spaces between words using MeCab."""
    m_result = mecab_tagger.parse(text).splitlines()
    m_result = m_result[:-1]  # Remove EOS
    words = []
    for v in m_result:
        if '\t' not in v: continue
        surface = v.split('\t')[0]
        words.append(surface)
    return ' '.join(words)

def get_pronunciation(text, mecab_tagger):
    """Get phonetic representation (katakana) of text."""
    m_result = mecab_tagger.parse(text).splitlines() 
    m_result = m_result[:-1] 
    pro = '' 
    for v in m_result:
        if '\t' not in v: continue
        surface = v.split('\t')[0] 
        p = v.split('\t')[1].split(',')[-1] 
        if p == '*': p = surface
        pro += p
    pro = jaconv.hira2kata(pro) 
    pro = format_text(pro) 
    return pro

def process_wiki_file(filepath, mecab_tagger):
    """Process a single wiki file to create input-output pairs."""
    pairs = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Original text with Kanji is the output
            kanji_text = line
            
            # Add spaces between words for better model training
            spaced_kanji = add_spaces(kanji_text, mecab_tagger)
            
            # Convert to phonetic representation (katakana) for input
            phonetic_text = get_pronunciation(kanji_text, mecab_tagger)
            
            # Add spaces to phonetic text as well
            spaced_phonetic = add_spaces(phonetic_text, mecab_tagger)
            
            # Create a pair
            pair = {
                "input": spaced_phonetic,
                "output": spaced_kanji
            }
            
            pairs.append(pair)
            
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
    
    return pairs

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Format Wikipedia articles for phonetic-to-Kanji conversion training')
    
    parser.add_argument('--input-dir', type=str, default='wiki_articles', 
                        help='Input directory containing Wikipedia articles')
    parser.add_argument('--output-dir', type=str, default='formatted_data', 
                        help='Output directory for formatted data')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of training data (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Ratio of validation data (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Ratio of test data (default: 0.1)')
    
    args = parser.parse_args()
    
    # Initialize MeCab with NEologd dictionary if available
    try:
        mecab_tagger = MeCab.Tagger('-r /dev/null -d /root/phonetic2kanji/abdp_ime/mecab-ipadic-neologd/lib')
    except:
        print("Warning: NEologd dictionary not found, using default dictionary")
        mecab_tagger = MeCab.Tagger('-r /dev/null')
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .txt files in the input directory and its subdirectories
    file_pattern = os.path.join(args.input_dir, '**', '*.txt')
    files = glob.glob(file_pattern, recursive=True)
    
    print(f"Found {len(files)} files to process")
    
    # Process all files
    all_pairs = []
    for filepath in tqdm(files, desc="Processing files"):
        pairs = process_wiki_file(filepath, mecab_tagger)
        all_pairs.extend(pairs)
    
    print(f"Created {len(all_pairs)} input-output pairs")
    
    # Shuffle the data
    import random
    random.shuffle(all_pairs)
    
    # Split the data
    n = len(all_pairs)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)
    
    train_pairs = all_pairs[:train_end]
    val_pairs = all_pairs[train_end:val_end]
    test_pairs = all_pairs[val_end:]
    
    print(f"Split data into {len(train_pairs)} training, {len(val_pairs)} validation, and {len(test_pairs)} test pairs")
    
    # Save the datasets
    train_path = output_dir / 'wiki_train.json'
    val_path = output_dir / 'wiki_val.json'
    test_path = output_dir / 'wiki_test.json'
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_pairs, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_pairs, f, ensure_ascii=False, indent=2)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"Formatted data saved to:")
    print(f"  Training: {train_path}")
    print(f"  Validation: {val_path}")
    print(f"  Test: {test_path}")

if __name__ == "__main__":
    main()
