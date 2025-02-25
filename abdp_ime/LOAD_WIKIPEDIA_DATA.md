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

The `wiki_dataprocessor.py` script converts the downloaded Wikipedia articles into paired .kana and .kanji files for phonetic-to-kanji conversion models:
- `.kana` files: Japanese text in hiragana form with spaces between characters
- `.kanji` files: Original text with kanji

#### Usage

Process all articles in all categories:
```
python preprocess/wiki_dataprocessor.py --input wiki_articles --output formatted_data
```

Process articles from a specific category:
```
python preprocess/wiki_dataprocessor.py --input wiki_articles --output formatted_data --category "物理学"
```

#### Command Line Arguments

- `--input`, `-i`: Path to the input directory containing article files (required)
- `--output`, `-o`: Directory to save output files (required)
- `--category`, `-c`: Specific category subfolder to process (optional)

#### Output Format

The script preserves the directory structure of the input and creates paired .kana and .kanji files:

```
formatted_data/
├── 物理学/
│ ├── 相対性理論.kana
│ ├── 相対性理論.kanji
│ ├── 量子力学.kana
│ ├── 量子力学.kanji
│ └── ...
├── 医学/
│ ├── 解剖学.kana
│ ├── 解剖学.kanji
│ └── ...
└── ...
```

Each line in the files is numbered and follows this format:

In .kana files:
```
1| こ れ は に ほ ん ご の ぶ ん し ょ う で す
```

In .kanji files:
```
1| これは日本語の文章です
```

#### Processing Steps

1. For each article file:
   - The original text with kanji is preserved in the .kanji file
   - Each character is converted to hiragana and spaces are added between characters in the .kana file
   - Line numbers are added to both files for easy reference

2. The directory structure from the input is maintained in the output.

#### Requirements

- jaconv (for hiragana conversion)
- tqdm (for progress bars)


## Example Usage of Entire Pipeline for General Domains

Source and download articles with sentences line-separated into `.txt` files:

```bash
python preprocess/wiki_dataloader.py --general-domains --max-pages 30 --output-dir wikipedia_articles
```

Process articles into `.kana` and `.kanji` file pairs:

```bash
python preprocess/wiki_dataprocessor.py -i wikipedia_articles -o wikipedia_formatted_data
```

resulting data is now compatible in pairs with `mecab_processor.py`.

