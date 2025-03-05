import MeCab
import jaconv
import unicodedata
import string
import os
import argparse
from tqdm import tqdm

# Function to format given text
def format_text(text):
    text = unicodedata.normalize("NFKC", text)  
    table = str.maketrans("", "", string.punctuation  + "「」、。・")
    text = text.translate(table)
    return text

m = MeCab.Tagger('-r /dev/null -d /root/phonetic2kanji/transformer_ime/mecab-ipadic-neologd/lib') 

# Function to create phonetics with spaces between hiragana
def getPronunciation(text):
    mecab_result = m.parse(text).splitlines() 
    mecab_result = mecab_result[:-1] 
    pronunciations = [] 
    for node in mecab_result:
        if '\t' not in node: continue
        surface = node.split('\t')[0] 
        reading = node.split('\t')[1].split(',')[-1] 
        if reading == '*': reading = surface
        pronunciations.append(reading)
        
    return ''.join(pronunciations)

def process_kanji_file(input_file, output_file, level):
    # Read the kanji file line by line
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line separately
    kana_lines = []
    for line in tqdm(lines, desc="Converting kanji to kana"):
        line = line.strip()  # Remove trailing whitespace
        if line:  # Skip empty lines
            kana = getPronunciation(line)
            kana_lines.append(kana)
    
    # If output_file is not specified, create it from input_file
    if output_file is None:
        output_file = input_file.replace('.kanji', '.kana')
    
    # Write the result to the .kana file, maintaining line structure
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(kana_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert kanji text to kana pronunciation')
    parser.add_argument('--input', type=str, default='data/data.kanji',
                        help='Input file path (default: data/data.kanji)')
    parser.add_argument('--output', type=str, default='data/data.kana',
                        help='Output file path (default: data/data.kana)')
    parser.add_argument('--level', type=str, default='default',
                        help='To split via character or via word ("char" or "word" or "default")')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    process_kanji_file(args.input, args.output, args.level)