import argparse
from pathlib import Path
import re
from tqdm import tqdm

def clean_and_filter_files(kana_file, output_file, kanji_file):
    # List of unnecessary characters to remove
    chars_to_remove_regex = r'[\ \　\.\,\‥\【\】\"\"\‼\⁉\−\〔\〕\、\。\〈\〉\《\》\「\」\『\』\・\！\（\）\，\．\？\［\］]'
    
    # Define the range of valid characters (katakana)
    MIN_CHAR = ord('ァ')  # Start of katakana range
    MAX_CHAR = ord('ー')  # End of katakana range
    
    def is_valid_kana(line):
        # Check if all characters are within valid range after cleaning
        for char in line:
            char_code = ord(char)
            if not (char == '\n' or MIN_CHAR <= char_code <= MAX_CHAR):
                return False
        return True
    
    # Count total lines for progress bar
    total_lines = sum(1 for _ in open(kana_file, 'r', encoding='utf-8'))
    
    # Process the files
    valid_lines = 0
    filtered_lines = 0
    
    # Create output paths
    output_kana = Path(output_file)
    output_kanji = output_kana.parent / (output_kana.stem + '.kanji')
    
    # Store valid lines in memory
    valid_kana_lines = []
    valid_kanji_lines = []
    
    with open(kana_file, 'r', encoding='utf-8') as f_kana, \
         open(kanji_file, 'r', encoding='utf-8') as f_kanji:
        
        for kana_line, kanji_line in tqdm(zip(f_kana, f_kanji), total=total_lines, desc="Processing lines"):
            # Remove special characters from both kana and kanji
            cleaned_kana = re.sub(chars_to_remove_regex, '', kana_line).strip()
            cleaned_kanji = re.sub(chars_to_remove_regex, '', kanji_line).strip()
            
            # Check if kana line contains only valid characters
            if cleaned_kana and is_valid_kana(cleaned_kana):
                valid_kana_lines.append(cleaned_kana)
                valid_kanji_lines.append(cleaned_kanji)
                valid_lines += 1
            else:
                filtered_lines += 1
    
    # Write filtered lines without final newline
    with open(output_kana, 'w', encoding='utf-8') as f_out_kana:
        f_out_kana.write('\n'.join(valid_kana_lines))
        
    with open(output_kanji, 'w', encoding='utf-8') as f_out_kanji:
        f_out_kanji.write('\n'.join(valid_kanji_lines))
    
    return valid_lines, filtered_lines

def main():
    parser = argparse.ArgumentParser(description="Clean and filter kana/kanji data files")
    # Input files
    parser.add_argument("--kana", required=True, 
                       help="Path to input kana file")
    parser.add_argument("--kanji", required=True,
                       help="Path to input kanji file")
    # Output files
    parser.add_argument("--clean_kana", required=True,
                       help="Path to output cleaned kana file")
    parser.add_argument("--clean_kanji", required=True,
                       help="Path to output cleaned kanji file")
    
    args = parser.parse_args()
    
    kana_path = Path(args.kana)
    kanji_path = Path(args.kanji)
    output_kana_path = Path(args.clean_kana)
    output_kanji_path = Path(args.clean_kanji)
    
    # Check if input files exist
    if not kana_path.exists():
        raise FileNotFoundError(f"Kana file not found: {kana_path}")
    if not kanji_path.exists():
        raise FileNotFoundError(f"Kanji file not found: {kanji_path}")
    
    # Create output directories if they don't exist
    output_kana_path.parent.mkdir(parents=True, exist_ok=True)
    output_kanji_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process the files
    valid_lines, filtered_lines = clean_and_filter_files(kana_path, output_kana_path, kanji_path)
    
    # Print statistics
    print("\nProcessing complete!")
    print(f"Valid lines: {valid_lines}")
    print(f"Filtered lines: {filtered_lines}")
    print(f"Total lines processed: {valid_lines + filtered_lines}")
    print(f"\nCleaned files saved to:")
    print(f"Kana: {output_kana_path}")
    print(f"Kanji: {output_kanji_path}")

if __name__ == "__main__":
    main()