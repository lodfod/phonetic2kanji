import argparse
import os
import re
from tqdm import tqdm
import MeCab

def convert_kanji_to_kana(mecab_path, kanji_text):
    """Convert kanji text to kana using MeCab."""
    try:
        if mecab_path:
            try:
                tagger = MeCab.Tagger(f'-r /dev/null -d {mecab_path}')
            except RuntimeError as e:
                print(f"Warning: Could not initialize MeCab with the provided path: {mecab_path}")
                print(f"Error details: {str(e)}")
                print("Falling back to default MeCab installation...")
                tagger = MeCab.Tagger("")
        else:
            # Use default MeCab installation
            tagger = MeCab.Tagger("")
    except RuntimeError as e:
        print(f"Error: Failed to initialize MeCab with both custom path and default installation.")
        print(f"Error details: {str(e)}")
        print("Please ensure MeCab is properly installed on your system.")
        raise RuntimeError("MeCab initialization failed. Cannot proceed with text conversion.")
    
    # Process using MeCab
    mecab_result = tagger.parse(kanji_text).splitlines()
    
    # Get phonetic pronounciation
    mecab_result = mecab_result[:-1] 
    pronunciations = [] 

    for node in mecab_result:
        if '\t' not in node: continue
        surface = node.split('\t')[0] 
        reading = node.split('\t')[1].split(',')[-1] 
        if reading == '*': reading = surface
        pronunciations.append(reading)
        
    return ''.join(pronunciations)

def process_file(input_file, output_file, mecab_path):
    """Process a file containing kanji text and convert to kana."""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    kana_lines = []
    for line in tqdm(lines, desc="Converting to kana"):
        kana_line = convert_kanji_to_kana(mecab_path, line)
        kana_lines.append(kana_line)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(kana_lines))
    
    print(f"Processed {len(lines)} lines.")
    print(f"Output saved to {output_file}")

def extract_wiki_text(input_file, max_sentences=None):
    """Extract text content from Wikipedia dump."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract article content
    articles = re.findall(r'<doc.*?>(.+?)</doc>', content, re.DOTALL)
    
    cleaned_articles = []
    for article in articles:
        # Remove headings, links, and other markup
        cleaned = re.sub(r'={2,}.*?={2,}', '', article)  # Remove headings
        cleaned = re.sub(r'\[\[.*?\]\]', '', cleaned)    # Remove links
        cleaned = re.sub(r'\{\{.*?\}\}', '', cleaned)    # Remove templates
        
        # Split into sentences and filter out short ones
        sentences = [s.strip() for s in re.split(r'[。．!！?？]', cleaned) if s.strip()]
        sentences = [s for s in sentences if len(s) > 10]  # Filter out short sentences
        
        cleaned_articles.extend(sentences)
    
    # Limit number of sentences if specified
    if max_sentences and len(cleaned_articles) > max_sentences:
        cleaned_articles = cleaned_articles[:max_sentences]
    
    return cleaned_articles

def main():
    parser = argparse.ArgumentParser(description="Process Japanese text for kana-kanji conversion")
    parser.add_argument("--input", required=True, help="Path to input file (kanji text)")
    parser.add_argument("--output", required=True, help="Path to output file (kana text)")
    parser.add_argument("--mecab_path", required=True, help="Path to MeCab dictionary")
    parser.add_argument("--wiki", action="store_true", help="Process input as Wikipedia dump")
    parser.add_argument("--max_sentences", type=int, default=None, help="Maximum number of sentences to process (for Wikipedia)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.wiki:
        # Process Wikipedia dump
        print("Extracting text from Wikipedia...")
        kanji_sentences = extract_wiki_text(args.input, args.max_sentences)
        
        # Save kanji sentences
        kanji_file = args.output.replace('.kana', '.kanji')
        with open(kanji_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(kanji_sentences))
        
        print(f"Extracted {len(kanji_sentences)} sentences.")
        print(f"Kanji data saved to {kanji_file}")
        
        # Convert to kana
        print("Converting to kana...")
        kana_lines = []
        for sentence in tqdm(kanji_sentences, desc="Converting to kana"):
            kana_line = convert_kanji_to_kana(args.mecab_path, sentence)
            kana_lines.append(kana_line)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(kana_lines))
        
        print(f"Kana data saved to {args.output}")
    else:
        # Process regular file
        process_file(args.input, args.output, args.mecab_path)

if __name__ == "__main__":
    main()
