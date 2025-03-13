import pykakasi
import mozcpy
import argparse
from pathlib import Path
from tqdm import tqdm

def katakana_to_hiragana(text):
    """Convert katakana text to hiragana using pykakasi"""
    # Initialize kakasi converter
    kakasi = pykakasi.kakasi()
    
    # Set mode to convert katakana to hiragana
    kakasi.setMode('K', 'H')  # Katakana to hiragana
    kakasi.setMode('H', 'H')  # Keep hiragana as is
    kakasi.setMode('J', 'H')  # Convert kanji to hiragana too (if any)
    
    converter = kakasi.getConverter()
    hiragana = converter.do(text)
    
    return hiragana

def process_kana_file(input_file, output_file):
    """Process a .kana file and output a .kanji file"""
    # Initialize mozcpy converter
    converter = mozcpy.Converter()
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read all lines from input file
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = [line.strip() for line in f_in.readlines()]
    
    # Process lines and collect results
    results = []
    for line in tqdm(lines, desc="Converting kana to kanji"):
        if not line:  # Skip empty lines
            results.append("")
            continue
            
        # Convert katakana to hiragana
        hiragana = katakana_to_hiragana(line)
        
        # Convert hiragana to kanji
        kanji = converter.convert(hiragana)
        
        # Add to results
        results.append(kanji)
    
    # Save to file with progress bar
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, kanji in enumerate(tqdm(results, desc="Writing to file", unit="line")):
            if i < len(results) - 1:
                f_out.write(f"{kanji}\n")
            else:
                f_out.write(f"{kanji}")  # No newline for the last sentence
    
    print(f"Processed {input_file} -> {output_file}")

def main():
    """Main function to parse arguments and process the file"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert kana file to kanji file')
    parser.add_argument('--kana_input', help='Input .kana file path')
    parser.add_argument('--kanji_output', help='Output .kanji file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the file
    process_kana_file(args.kana_input, args.kanji_output)

if __name__ == "__main__":
    main()