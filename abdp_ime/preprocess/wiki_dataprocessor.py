import os
import re
import json
import unicodedata
import jaconv
import argparse
from tqdm import tqdm

class DataProcessor:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def process_data(self):
        """Process the input data and create .kana and .kanji files."""
        # Read the input data (text file with Japanese text)
        data = self._read_input_data()
        
        # Process the data into kana and kanji pairs
        kana_lines = []
        kanji_lines = []
        
        for i, line in enumerate(tqdm(data, desc="Processing data")):
            # Skip empty lines
            if not line.strip():
                continue
                
            # Create the kanji version (original text)
            kanji_text = line.strip()
            
            # Create the kana version (convert to hiragana with spaces)
            kana_text = self._convert_to_spaced_kana(kanji_text)
            
            # Add to the respective lists
            kana_lines.append(kana_text)
            kanji_lines.append(kanji_text)
        
        # Write the output files
        self._write_output_files(kana_lines, kanji_lines)
        
    def _read_input_data(self):
        """Read the input data from the specified path."""
        # Read text file with Japanese text
        with open(self.input_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _convert_to_spaced_kana(self, text):
        """Convert text to hiragana with spaces between characters."""
        # Convert to hiragana
        hiragana = jaconv.kata2hira(text)
        
        # Add spaces between Japanese characters
        spaced_hiragana = ' '.join(c for c in hiragana if not c.isspace())
        
        return spaced_hiragana
    
    def _write_output_files(self, kana_lines, kanji_lines):
        """Write the processed data to .kana and .kanji files."""
        # Generate base filename from the input path
        base_filename = os.path.splitext(os.path.basename(self.input_path))[0]
        
        # Write the kana file
        kana_path = os.path.join(self.output_dir, f"{base_filename}.kana")
        with open(kana_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(kana_lines, 1):
                f.write(f"{i}| {line}\n")
        
        # Write the kanji file
        kanji_path = os.path.join(self.output_dir, f"{base_filename}.kanji")
        with open(kanji_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(kanji_lines, 1):
                f.write(f"{i}| {line}\n")
        
        print(f"Created {kana_path} and {kanji_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process Japanese text into kana and kanji files.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input text file with Japanese text')
    parser.add_argument('--output', '-o', required=True, help='Directory to save output files')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    processor = DataProcessor(
        input_path=args.input,
        output_dir=args.output
    )
    processor.process_data()
