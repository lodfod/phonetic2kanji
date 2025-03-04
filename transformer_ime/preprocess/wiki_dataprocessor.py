import argparse
import os
import re
from tqdm import tqdm
import MeCab

def extract_wiki_text(input_file):
    """Extract text content from Wikipedia dump."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract article content (simplified - you might need to adjust this)
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
    
    return cleaned_articles

def create_kana_from_kanji(mecab_path, kanji_sentences):
    """Convert kanji sentences to kana using MeCab."""
    tagger = MeCab.Tagger(f"-d {mecab_path}")
    
    kana_sentences = []
    for sentence in tqdm(kanji_sentences, desc="Converting to kana"):
        words = []
        node = tagger.parseToNode(sentence)
        
        while node:
            if node.surface:  # Skip empty nodes
                features = node.feature.split(',')
                if len(features) > 7:
                    # Get reading (kana) if available
                    kana = features[7]
                    if kana == '*':
                        # Use surface form if no reading is available
                        words.append(node.surface)
                    else:
                        words.append(kana)
                else:
                    words.append(node.surface)
            node = node.next
        
        kana_sentence = ''.join(words)
        kana_sentences.append(kana_sentence)
    
    return kana_sentences

def main():
    parser = argparse.ArgumentParser(description="Process Wikipedia data for kana-kanji conversion")
    parser.add_argument("--wiki_file", required=True, help="Path to Wikipedia dump file")
    parser.add_argument("--mecab_path", required=True, help="Path to MeCab dictionary")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed data")
    parser.add_argument("--max_sentences", type=int, default=100000, help="Maximum number of sentences to process")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract text from Wikipedia
    print("Extracting text from Wikipedia...")
    kanji_sentences = extract_wiki_text(args.wiki_file)
    
    # Limit number of sentences if needed
    if len(kanji_sentences) > args.max_sentences:
        print(f"Limiting to {args.max_sentences} sentences...")
        kanji_sentences = kanji_sentences[:args.max_sentences]
    
    # Create kana from kanji
    print("Creating kana from kanji...")
    kana_sentences = create_kana_from_kanji(args.mecab_path, kanji_sentences)
    
    # Save to files
    kanji_file = os.path.join(args.output_dir, "wiki.kanji")
    kana_file = os.path.join(args.output_dir, "wiki.kana")
    
    with open(kanji_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(kanji_sentences))
    
    with open(kana_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(kana_sentences))
    
    print(f"Processed {len(kanji_sentences)} sentences.")
    print(f"Kanji data saved to {kanji_file}")
    print(f"Kana data saved to {kana_file}")

if __name__ == "__main__":
    main() 