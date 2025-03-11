import argparse
from pathlib import Path
import re
from tqdm import tqdm
import unicodedata

def clean_and_filter_files(kana_file, output_file, kanji_file):
    # List of unnecessary characters to remove
    chars_to_remove_regex = r'[\.\,\‥\【\】\"\"\‼\⁉\−\〔\〕\、\。\〈\〉\《\》\「\」\『\』\・\！\（\）\，\．\？\［\］]'
    
    # Define the range of valid characters (katakana)
    MIN_CHAR = ord('ァ')  # Start of katakana range
    MAX_CHAR = ord('ー')  # End of katakana range
    
    # Regex patterns for filtering
    url_pattern = r'https?://\S+'
    isbn_pattern = r'ISBN\s+[\d\-]+'
    email_pattern = r'\S+@\S+\.\S+'
    english_word_pattern = r'[a-zA-Z]+'
    number_sequence_pattern = r'\d+'
    
    def is_valid_kana(line):
        # Check if all characters are within valid range after cleaning
        for char in line:
            char_code = ord(char)
            if not (char == '\n' or MIN_CHAR <= char_code <= MAX_CHAR):
                return False
        return True
    
    def is_kanji(char):
        # Check if a character is a kanji
        return 'CJK UNIFIED IDEOGRAPH' in unicodedata.name(char, '')
    
    def is_hiragana(char):
        # Check if a character is hiragana
        return ord('ぁ') <= ord(char) <= ord('ゖ')
    
    def is_katakana(char):
        # Check if a character is katakana
        return ord('ァ') <= ord(char) <= ord('ヶ')
    
    def is_japanese_punctuation(char):
        # Common Japanese punctuation
        jpn_punct = '　、。，．・：；？！゛゜´｀¨＾￣＿ヽヾゝゞ〃仝々〆〇ー―‐／＼～∥｜…‥''""（）〔〕［］｛｝〈〉《》「」『』【】＋－±×÷＝≠＜＞≦≧∞∴♂♀°′″℃￥＄￠￡％＃＆＊＠§☆★○●◎◇◆□■△▲▽▼※〒→←↑↓〓'
        return char in jpn_punct
    
    def is_acceptable_in_japanese_text(char):
        # Check if a character is acceptable in Japanese text
        # This includes kanji, hiragana, katakana, Japanese punctuation, 
        # digits, and some common symbols
        return (is_kanji(char) or 
                is_hiragana(char) or 
                is_katakana(char) or 
                is_japanese_punctuation(char) or
                char.isdigit() or
                char in ',.!?;:()[]{} ')
    
    def clean_kanji_line(line):
        # Remove URLs
        line = re.sub(url_pattern, '', line)
        
        # Remove ISBN numbers
        line = re.sub(isbn_pattern, '', line)
        
        # Remove email addresses
        line = re.sub(email_pattern, '', line)
        
        # Remove English words
        line = re.sub(english_word_pattern, '', line)
        
        # Remove number sequences (optional, might want to keep some numbers)
        # line = re.sub(number_sequence_pattern, '', line)
        
        # Remove special characters
        line = re.sub(chars_to_remove_regex, '', line)
        
        # Filter out characters that aren't acceptable in Japanese text
        filtered_chars = [char for char in line if is_acceptable_in_japanese_text(char)]
        
        return ''.join(filtered_chars).strip()
    
    # Read all lines from both files
    with open(kana_file, 'r', encoding='utf-8') as f_kana, \
         open(kanji_file, 'r', encoding='utf-8') as f_kanji:
        kana_lines = [line.strip() for line in f_kana]
        kanji_lines = [line.strip() for line in f_kanji]
    
    # Filter out blank lines from both files
    non_blank_pairs = []
    blank_lines_removed = 0
    
    for kana_line, kanji_line in zip(kana_lines, kanji_lines):
        if kana_line and kanji_line:  # Both lines must be non-empty
            non_blank_pairs.append((kana_line, kanji_line))
        else:
            blank_lines_removed += 1
    
    # Process the non-blank lines
    valid_lines = 0
    filtered_lines = 0
    
    # Create output paths
    output_kana = Path(output_file)
    output_kanji = output_kana.parent / (output_kana.stem + '.kanji')
    
    # Store valid lines in memory
    valid_kana_lines = []
    valid_kanji_lines = []
    
    for kana_line, kanji_line in tqdm(non_blank_pairs, desc="Processing lines"):
        # Clean the kana line (remove special characters)
        cleaned_kana = re.sub(chars_to_remove_regex, '', kana_line).strip()
        
        # Clean the kanji line (remove non-Japanese content)
        cleaned_kanji = clean_kanji_line(kanji_line)
        
        # Check if kana line contains only valid characters and both lines are non-empty after cleaning
        if cleaned_kana and cleaned_kanji and is_valid_kana(cleaned_kana):
            valid_kana_lines.append(cleaned_kana)
            valid_kanji_lines.append(cleaned_kanji)
            valid_lines += 1
        else:
            filtered_lines += 1
    
    # Verify that we have the same number of lines in both files
    assert len(valid_kana_lines) == len(valid_kanji_lines), "Line count mismatch after filtering"
    
    # Write filtered lines without final newline
    with open(output_kana, 'w', encoding='utf-8') as f_out_kana:
        f_out_kana.write('\n'.join(valid_kana_lines))
        
    with open(output_kanji, 'w', encoding='utf-8') as f_out_kanji:
        f_out_kanji.write('\n'.join(valid_kanji_lines))
    
    return valid_lines, filtered_lines, blank_lines_removed

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
    # Additional options
    parser.add_argument("--keep_numbers", action="store_true", default=False,
                       help="Keep numeric sequences in the output")
    
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
    valid_lines, filtered_lines, blank_lines_removed = clean_and_filter_files(kana_path, output_kana_path, kanji_path)
    
    # Print statistics
    print("\nProcessing complete!")
    print(f"Valid lines: {valid_lines}")
    print(f"Filtered lines: {filtered_lines}")
    print(f"Blank lines removed: {blank_lines_removed}")
    print(f"Total lines processed: {valid_lines + filtered_lines + blank_lines_removed}")
    print(f"\nCleaned files saved to:")
    print(f"Kana: {output_kana_path}")
    print(f"Kanji: {output_kanji_path}")
    
    # Verify line counts match
    with open(output_kana_path, 'r', encoding='utf-8') as f_kana, \
         open(output_kanji_path, 'r', encoding='utf-8') as f_kanji:
        kana_count = sum(1 for _ in f_kana)
        kanji_count = sum(1 for _ in f_kanji)
    
    print(f"\nVerification:")
    print(f"Output kana file line count: {kana_count}")
    print(f"Output kanji file line count: {kanji_count}")
    print(f"Line counts match: {kana_count == kanji_count}")

if __name__ == "__main__":
    main()