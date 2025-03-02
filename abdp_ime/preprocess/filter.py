import argparse
from tqdm import tqdm

def filter_files(kanji_path, kana_path, max_len):
    # Read both files
    with open(kana_path, 'r', encoding='utf-8') as f:
        kana_lines = f.readlines()
    with open(kanji_path, 'r', encoding='utf-8') as f:
        kanji_lines = f.readlines()
    
    # Ensure both files have same number of lines
    assert len(kana_lines) == len(kanji_lines), "Kana and Kanji files must have same number of lines"
    
    # Filter lines
    filtered_kana = []
    filtered_kanji = []
    
    for kana, kanji in tqdm(zip(kana_lines, kanji_lines), total=len(kana_lines), desc="Filtering lines"):
        # Strip whitespace
        kana = kana.strip()
        kanji = kanji.strip()
        
        # Check length
        if len(kana) <= max_len:
            filtered_kana.append(kana + '\n')
            filtered_kanji.append(kanji + '\n')
    
    # Write filtered content back to files
    with open(kana_path, 'w', encoding='utf-8') as f:
        f.writelines(filtered_kana[:-1])  # Remove last newline
        f.write(filtered_kana[-1].rstrip())  # Write last line without newline
    with open(kanji_path, 'w', encoding='utf-8') as f:
        f.writelines(filtered_kanji[:-1])  # Remove last newline
        f.write(filtered_kanji[-1].rstrip())  # Write last line without newline
    
    print(f"Removed {len(kana_lines) - len(filtered_kana)} lines that exceeded {max_len} characters")

def main():
    parser = argparse.ArgumentParser(description='Filter lines from kana/kanji files based on character length')
    parser.add_argument('--kanji_path', required=True, help='Path to the kanji file')
    parser.add_argument('--kana_path', required=True, help='Path to the kana file')
    parser.add_argument('--max_len', type=int, required=True, help='Maximum allowed character length')
    
    args = parser.parse_args()
    
    filter_files(args.kanji_path, args.kana_path, args.max_len)

if __name__ == '__main__':
    main()
