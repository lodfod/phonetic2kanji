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

