from datasets import load_dataset
import MeCab
import jaconv
import unicodedata
import string
from tqdm import tqdm
import argparse
from pathlib import Path

# Load reazonspeech corpus dataset
ds = load_dataset("reazon-research/reazonspeech", trust_remote_code=True)
print("number of rows: ", ds["train"].num_rows)

# Print some dataset
print("Example of data...")
for i in range(10):
    print(f"row {i}: {ds['train'][i]['transcription']}")

# Function to format given text
def format_text(text):
    text = unicodedata.normalize("NFKC", text)  
    table = str.maketrans("", "", string.punctuation  + "「」、。・")
    text = text.translate(table)
    return text

# Function to add spaces between words
def add_spaces(text):
    m_result = m.parse(text).splitlines()
    m_result = m_result[:-1]  # Remove EOS
    words = []
    for v in m_result:
        if '\t' not in v: continue
        surface = v.split('\t')[0]
        words.append(surface)
    return ' '.join(words)

m = MeCab.Tagger('-r /dev/null -d /root/phonetic2kanji/abdp_ime/mecab-ipadic-neologd/lib')

# Function to create phonetics
def getPronunciation(text):
    m_result = m.parse(text).splitlines() 
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process ReazonSpeech dataset and save transcriptions')
    parser.add_argument('--output', type=str, default='data/data.kanji',
                       help='Output file path (default: data/data.kanji)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save transcriptions to output file with progress bar
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(ds['train'], desc="Saving transcriptions"):
            spaced_text = add_spaces(item['transcription'])
            f.write(spaced_text + '\n')

if __name__ == '__main__':
    main()