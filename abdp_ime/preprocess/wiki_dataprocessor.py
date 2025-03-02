import os
import re
import json
import unicodedata
import jaconv
import argparse
import glob
from tqdm import tqdm

class DataProcessor:
    def __init__(self, input_path, output_dir, category=None):
        """
        Initialize the data processor.
        
        Args:
            input_path (str): Path to the input directory containing article files
            output_dir (str): Directory to save output files
            category (str, optional): Specific category subfolder to process
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.category = category
        os.makedirs(output_dir, exist_ok=True)
        
    def process_data(self):
        """Process the input data and create .kana and .kanji files."""
        # Get all files to process based on category
        files_to_process = self._get_files_to_process()
        
        if not files_to_process:
            print(f"No files found to process in {self.input_path}")
            return
            
        print(f"Found {len(files_to_process)} files to process")
        
        # Process each file
        for file_path in tqdm(files_to_process, desc="Processing files"):
            self._process_single_file(file_path)
    
    def _get_files_to_process(self):
        """Get all files to process based on category."""
        if self.category:
            # Process files in the specific category subfolder
            category_path = os.path.join(self.input_path, self.category)
            if not os.path.exists(category_path):
                print(f"Category folder not found: {category_path}")
                return []
            files = glob.glob(os.path.join(category_path, "*.txt"))
        else:
            # Process all .txt files in the input directory and its subdirectories
            files = []
            for root, _, filenames in os.walk(self.input_path):
                for filename in filenames:
                    if filename.endswith(".txt"):
                        files.append(os.path.join(root, filename))
        
        return files
    
    def _process_single_file(self, file_path):
        """Process a single file and create .kana and .kanji files."""
        # Read the input data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [line.strip() for line in f if line.strip()]
        
        # Process the data into kana and kanji pairs
        kana_lines = []
        kanji_lines = []
        
        for line in data:
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
        self._write_output_files(kana_lines, kanji_lines, file_path)
    
    def _convert_to_spaced_kana(self, text):
        """Convert text to hiragana with spaces between characters."""
        # Convert to hiragana
        hiragana = jaconv.kata2hira(text)
        
        # Add spaces between Japanese characters
        spaced_hiragana = ' '.join(c for c in hiragana if not c.isspace())
        
        return spaced_hiragana
    
    def _write_output_files(self, kana_lines, kanji_lines, original_file_path):
        """Write the processed data to .kana and .kanji files."""
        # Preserve the directory structure relative to input_path
        rel_path = os.path.relpath(original_file_path, self.input_path)
        dir_path = os.path.dirname(rel_path)
        
        # Create the output directory structure if needed
        output_subdir = os.path.join(self.output_dir, dir_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Generate base filename from the original file path
        base_filename = os.path.splitext(os.path.basename(original_file_path))[0]
        
        # Write the kana file
        kana_path = os.path.join(output_subdir, f"{base_filename}.kana")
        with open(kana_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(kana_lines, 1):
                f.write(f"{i}| {line}\n")
        
        # Write the kanji file
        kanji_path = os.path.join(output_subdir, f"{base_filename}.kanji")
        with open(kanji_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(kanji_lines, 1):
                f.write(f"{i}| {line}\n")
        
        print(f"Created {kana_path} and {kanji_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process Japanese text into kana and kanji files.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input directory containing article files')
    parser.add_argument('--output', '-o', required=True, help='Directory to save output files')
    parser.add_argument('--category', '-c', help='Specific category subfolder to process')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    processor = DataProcessor(
        input_path=args.input,
        output_dir=args.output,
        category=args.category
    )
    processor.process_data()
