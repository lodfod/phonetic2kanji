# Wikipedia Data Loader

A tool for downloading and processing Wikipedia articles in Japanese, with one sentence per line.

## Usage

### Basic Usage

Download articles from a specific category:
```
python wiki_dataloader.py --category "物理学" --max-pages 50 --output-dir wiki_data
```

Download specific articles by title:
```
python wiki_dataloader.py --titles "東京" "大阪" "京都" --output-dir wiki_data
```

Download articles from general domains (Physics, Medical, Tech, etc.):
```
python wiki_dataloader.py --general-domains --max-pages 30 --output-dir wiki_data
```

Discover popular categories and download articles from them:
```
python wiki_dataloader.py --discover-categories --num-categories 10 --max-pages 20 --output-dir wiki_data
```

Download from categories listed in a file:
```
python wiki_dataloader.py --categories-file my_categories.txt --max-pages 50 --output-dir wiki_data
```

### Command Line Arguments

- `--category`: Category to download articles from
- `--titles`: Specific article titles to download
- `--output-dir`: Output directory (default: 'wiki_articles')
- `--language`: Wikipedia language code (default: 'ja' for Japanese)
- `--max-depth`: Maximum category recursion depth (default: 1)
- `--max-pages`: Maximum number of pages to download per category (default: 100)
- `--popular`: Download articles from popular Japanese categories
- `--num-categories`: Number of popular categories to use (default: 10)
- `--categories-file`: File containing category names, one per line
- `--discover-categories`: Discover popular categories and download articles from them
- `--seed-categories`: Seed categories for discovery (used with --discover-categories)
- `--general-domains`: Download articles from general domains in Japanese

## Output Format

The script creates a directory structure organized by category:

```
wiki_articles/
├── 物理学/
│ ├── 相対性理論.txt
│ ├── 量子力学.txt
│ └── ...
├── 医学/
│ ├── 解剖学.txt
│ ├── 免疫学.txt
│ └── ...
└── ...
```

Each article is saved as a text file with one sentence per line.

## Notes

- The script includes rate limiting to avoid overwhelming the Wikipedia API
- Japanese sentence tokenization is handled automatically
- All text is saved in UTF-8 encoding


## Data Processing

After downloading Wikipedia articles, you can process them for phonetic-to-kanji conversion training using the `wiki_dataprocessor.py` script.

### Wiki Data Processor

The `wiki_dataprocessor.py` script converts the downloaded Wikipedia articles into training data for phonetic-to-kanji conversion models. It creates input-output pairs where:
- Input: Japanese text in phonetic form (katakana with spaces between words)
- Output: Original text with kanji (with spaces between words)

#### Usage

```
python preprocess/wiki_dataprocessor.py --input-dir wiki_articles --output-dir formatted_data
```


#### Command Line Arguments

- `--input-dir`: Input directory containing Wikipedia articles (default: 'wiki_articles')
- `--output-dir`: Output directory for formatted data (default: 'formatted_data')
- `--train-ratio`: Ratio of training data (default: 0.8)
- `--val-ratio`: Ratio of validation data (default: 0.1)
- `--test-ratio`: Ratio of test data (default: 0.1)

#### Output Format

The script produces three JSON files:
- `wiki_train.json`: Training dataset
- `wiki_val.json`: Validation dataset
- `wiki_test.json`: Test dataset

Each file contains a list of input-output pairs in the following format:

```
[
{
"input": "カガク ノ リロン",
"output": "科学 の 理論"
},
...
]
```


#### Processing Steps

1. For each sentence in the Wikipedia articles:
   - The original text with kanji is preserved as the output
   - MeCab is used to get the phonetic representation (katakana) as the input
   - Spaces are added between words in both input and output for better model training
   - Punctuation is removed and text is normalized

2. The data is shuffled and split into training, validation, and test sets according to the specified ratios.

#### Requirements

- MeCab with NEologd dictionary (falls back to default dictionary if not available)
- jaconv (for hiragana to katakana conversion)
- tqdm (for progress bars)



