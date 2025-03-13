import os
import wikipediaapi
import argparse
import time
import re
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize


domain_map = {
            "Physics": ["物理学", "物理現象", "力学", "電磁気学", "量子力学", "相対性理論"],
            "Medical": ["医学", "医療", "疾患", "解剖学", "薬学", "公衆衛生学"],
            "Technology": ["技術", "工学", "コンピュータ", "ソフトウェア", "人工知能", "ロボット工学"],
            "Literature": ["文学", "小説", "詩", "文学作品", "文学理論", "作家"],
            "History": ["歴史", "古代史", "中世史", "近代史", "考古学", "文明"],
            "Companies": ["企業", "会社", "ビジネス", "多国籍企業", "スタートアップ企業"],
            "Finance": ["金融", "経済学", "投資", "銀行", "株式市場", "保険"],
            "Industry": ["産業", "製造業", "農業", "鉱業", "エネルギー産業", "自動車産業"],
            "Arts": ["芸術", "絵画", "彫刻", "建築", "音楽", "演劇", "映画"]
        }

class WikipediaDataLoader:
    def __init__(self, language='ja', output_dir=None):
        """
        Initialize the Wikipedia data loader.
        
        Args:
            language (str): Language code for Wikipedia (default: 'ja' for Japanese)
            output_dir (str): Directory to save the downloaded articles (optional)
        """
        self.language = language
        self.output_dir = output_dir
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='WikipediaDataLoader/1.0 (your-email@example.com)',
            language=language
        )
        
        # Create output directory if it doesn't exist and is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Download Japanese sentence tokenizer if needed
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            print("Error downloading NLTK punkt. Will use regex-based sentence splitting as fallback")

    def get_category_members(self, category_name, max_depth=1, max_pages=100):
        """
        Get all pages from a category recursively.
        
        Args:
            category_name (str): Name of the category
            max_depth (int): Maximum depth for category recursion
            max_pages (int): Maximum number of pages to retrieve
            
        Returns:
            list: List of page titles
        """
        category = self.wiki.page(f"Category:{category_name}")
        if not category.exists():
            tqdm.write(f"Category '{category_name}' does not exist")
            return []
        
        pages = []
        self._get_category_members_recursive(
            category.categorymembers, 
            pages, 
            level=0, 
            max_level=max_depth, 
            max_pages=max_pages
        )
        return pages

    def _get_category_members_recursive(self, categorymembers, pages, level=0, max_level=1, max_pages=100):
        """
        Helper method to recursively get category members.
        
        Args:
            categorymembers (dict): Dictionary of category members
            pages (list): List to store page titles
            level (int): Current recursion level
            max_level (int): Maximum recursion level
            max_pages (int): Maximum number of pages to retrieve
        """
        for page_title, page in categorymembers.items():
            if len(pages) >= max_pages:
                return
                
            if page.ns == wikipediaapi.Namespace.MAIN:
                pages.append(page)
                
            if page.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
                self._get_category_members_recursive(
                    page.categorymembers, 
                    pages, 
                    level=level+1, 
                    max_level=max_level,
                    max_pages=max_pages
                )

    def download_article(self, page_title):
        """
        Download a single Wikipedia article.
        
        Args:
            page_title (str): Title of the Wikipedia page
            
        Returns:
            str: Text of the article with one sentence per line
        """
        page = self.wiki.page(page_title)
        if not page.exists():
            return None
        
        # Get the full text of the article
        text = page.text
        
        # Try to use NLTK's sentence tokenizer
        try:
            sentences = sent_tokenize(text, language='japanese')
        except LookupError:
            # Fallback to simple regex-based sentence splitting
            # Japanese sentences typically end with one of these characters
            sentences = re.split(r'([。！？])', text)
            # Recombine the sentences with their punctuation
            processed_sentences = []
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    processed_sentences.append(sentences[i] + sentences[i+1])
                else:
                    processed_sentences.append(sentences[i])
            sentences = processed_sentences
        
        # Clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            # Remove excessive whitespace
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            if sentence:  # Only add non-empty sentences
                cleaned_sentences.append(sentence)
                
        return '\n'.join(cleaned_sentences)

    def download_articles_from_category(self, category_name, max_depth=1, max_pages=100, output_file=None):
        """
        Download all articles from a category and save them to a single .kanji file.
        
        Args:
            category_name (str): Name of the category
            max_depth (int): Maximum depth for category recursion
            max_pages (int): Maximum number of pages to retrieve
            output_file (str): Path to the output file (if None, will be generated based on category name)
            
        Returns:
            str: Path to the saved .kanji file
        """
        pages = self.get_category_members(category_name, max_depth, max_pages)
        
        # Determine the output filepath
        if output_file:
            filepath = output_file
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        else:
            # Create a safe filename from the category name
            safe_category = re.sub(r'[^\w\s-]', '', category_name).strip().replace(' ', '_')
            filepath = os.path.join(self.output_dir, f"{safe_category}.kanji")
        
        # Collect all article texts
        all_articles_text = []
        
        for page in tqdm(pages, desc=f"Downloading articles from {category_name}", leave=False):
            # Get article text with one sentence per line
            article_text = self.download_article(page.title)
            
            if article_text:
                # Add article title as a header
                article_with_header = f"# {page.title}\n{article_text}\n\n"
                all_articles_text.append(article_with_header)
            else:
                tqdm.write(f"Failed to download article: {page.title}")
            
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.5)
        
        # Save all articles to a single .kanji file
        if all_articles_text:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(''.join(all_articles_text))
            tqdm.write(f"✓ Saved {len(all_articles_text)} articles to {filepath}")
            return filepath
        else:
            tqdm.write(f"✗ No articles downloaded for category: {category_name}")
            return None

    def download_articles_from_multiple_categories(self, category_names, max_depth=1, max_pages_per_category=100):
        """
        Download articles from multiple categories, each to its own .kanji file.
        
        Args:
            category_names (list): List of category names
            max_depth (int): Maximum depth for category recursion
            max_pages_per_category (int): Maximum number of pages to retrieve per category
            
        Returns:
            list: List of paths to saved .kanji files
        """
        all_saved_files = []
        
        for category_name in tqdm(category_names, desc="Processing categories"):
            tqdm.write(f"Processing category: {category_name}")
            saved_file = self.download_articles_from_category(
                category_name, 
                max_depth=max_depth,
                max_pages=max_pages_per_category
            )
            if saved_file:
                all_saved_files.append(saved_file)
            
            # Add a delay between categories to avoid hitting rate limits
            time.sleep(2)
        
        return all_saved_files

    def download_articles_from_titles(self, titles):
        """
        Download articles from a list of titles and save to a single .kanji file.
        
        Args:
            titles (list): List of page titles
            
        Returns:
            str: Path to the saved .kanji file
        """
        filepath = os.path.join(self.output_dir, "custom_articles.kanji")
        all_articles_text = []
        
        for title in tqdm(titles, desc="Downloading articles"):
            page = self.wiki.page(title)
            if page.exists():
                article_text = self.download_article(title)
                if article_text:
                    # Add article title as a header
                    article_with_header = f"# {title}\n{article_text}\n\n"
                    all_articles_text.append(article_with_header)
                else:
                    tqdm.write(f"Failed to download article: {title}")
            else:
                tqdm.write(f"Page '{title}' does not exist")
                
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.5)
        
        # Save all articles to a single .kanji file
        if all_articles_text:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(''.join(all_articles_text))
            tqdm.write(f"✓ Saved {len(all_articles_text)} articles to {filepath}")
            return filepath
        else:
            tqdm.write(f"✗ No articles downloaded")
            return None

    def find_general_domain_categories(self):
        """
        Returns a list of general domain categories in Japanese Wikipedia.
        
        Returns:
            dict: Dictionary mapping domain names to lists of category names
        """
        # Map of general domains to their Japanese Wikipedia category names
        general_domains = domain_map
        print(f"Using {sum(len(cats) for cats in general_domains.values())} categories across {len(general_domains)} domains")
        return general_domains

    def download_articles_from_domain(self, domain_name, max_depth=1, max_pages_per_category=20, output_file=None):
        """
        Download articles from all categories associated with a domain.
        
        Args:
            domain_name (str): Name of the domain (e.g., "Technology")
            max_depth (int): Maximum depth for category recursion
            max_pages_per_category (int): Maximum number of pages to retrieve per category
            output_file (str): Path to the output file (if None, will be generated based on domain name)
            
        Returns:
            str: Path to the saved .kanji file
        """
        # Get the domain categories mapping
        domain_categories = self.find_general_domain_categories()
        
        # Check if the domain exists in our mapping
        domain_key = None
        for key in domain_categories.keys():
            if key.lower() == domain_name.lower():
                domain_key = key
                break
        
        if not domain_key:
            tqdm.write(f"Domain '{domain_name}' not found in domain map. Available domains: {list(domain_categories.keys())}")
            return None
        
        categories = domain_categories[domain_key]
        tqdm.write(f"Processing domain: {domain_key} with categories: {categories}")
        
        # Determine the output filepath
        if output_file:
            filepath = output_file
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        else:
            # Create a safe filename from the domain name
            safe_domain = re.sub(r'[^\w\s-]', '', domain_name).strip().replace(' ', '_')
            filepath = os.path.join(self.output_dir, f"{safe_domain}.kanji")
        
        # Collect all articles from all categories in this domain
        all_domain_articles = []
        
        # Process each category in this domain
        for category in tqdm(categories, desc=f"Categories in {domain_key}", leave=False):
            tqdm.write(f"  Category: {category}")
            
            # Get pages from this category
            pages = self.get_category_members(category, max_depth, max_pages_per_category)
            
            # Download each article
            for page in tqdm(pages, desc=f"Articles in {category}", leave=False):
                # Get article text with one sentence per line
                article_text = self.download_article(page.title)
                
                if article_text:
                    # Add article title and category as a header
                    article_with_header = f"# {page.title} (Category: {category})\n{article_text}\n\n"
                    all_domain_articles.append(article_with_header)
                else:
                    tqdm.write(f"Failed to download article: {page.title}")
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(0.2)
        
        # Save all articles to a single .kanji file
        if all_domain_articles:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(''.join(all_domain_articles))
            tqdm.write(f"✓ Saved {len(all_domain_articles)} articles to {filepath}")
            return filepath
        else:
            tqdm.write(f"✗ No articles downloaded for domain: {domain_name}")
            return None

    def download_general_domain_articles(self, max_depth=1, max_pages_per_category=20, output_dir=None):
        """
        Download articles from general domains in Japanese, with one file per domain.
        
        Args:
            max_depth (int): Maximum depth for category recursion
            max_pages_per_category (int): Maximum number of pages to retrieve per category
            output_dir (str): Directory to save the output files (overrides self.output_dir)
            
        Returns:
            list: List of paths to saved files
        """
        domain_categories = self.find_general_domain_categories()
        saved_files = []
        
        # Process each domain
        for domain_name in tqdm(domain_categories.keys(), desc="Processing domains"):
            # Determine output file path
            if output_dir:
                safe_domain = re.sub(r'[^\w\s-]', '', domain_name).strip().replace(' ', '_')
                output_file = os.path.join(output_dir, f"{safe_domain}.kanji")
            else:
                output_file = None
                
            saved_file = self.download_articles_from_domain(
                domain_name,
                max_depth=max_depth,
                max_pages_per_category=max_pages_per_category,
                output_file=output_file
            )
            if saved_file:
                saved_files.append(saved_file)
        
        return saved_files

def main():
    parser = argparse.ArgumentParser(description='Download Wikipedia articles with one sentence per line')
    
    parser.add_argument('--category', type=str, help='Category/domain to download articles from (e.g., Technology, Literature, Arts)')
    parser.add_argument('--titles', type=str, nargs='+', help='Specific article titles to download')
    parser.add_argument('--output-dir', type=str, default='wiki_articles', help='Output directory (used if --output-file not specified)')
    parser.add_argument('--output-file', type=str, help='Output file path (overrides --output-dir)')
    parser.add_argument('--language', type=str, default='ja', help='Wikipedia language code')
    parser.add_argument('--max-depth', type=int, default=1, help='Maximum category recursion depth')
    parser.add_argument('--max-pages', type=int, default=100, help='Maximum number of pages to download')
    parser.add_argument('--num-categories', type=int, default=10, help='Number of popular categories to use')
    parser.add_argument('--categories-file', type=str, help='File containing category names, one per line')
    parser.add_argument('--general-domains', action='store_true', 
                        help='Download articles from general domains (Physics, Medical, Tech, etc.) in Japanese')
    parser.add_argument('--discover-categories', action='store_true',
                        help='Discover and use popular categories automatically')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist and is specified
    if args.output_dir and not args.output_file:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # If output_file is specified, ensure its directory exists
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    loader = WikipediaDataLoader(language=args.language, output_dir=args.output_dir)
    
    if args.general_domains:
        saved_files = loader.download_general_domain_articles(
            max_depth=args.max_depth,
            max_pages_per_category=args.max_pages,
            output_dir=args.output_dir
        )
        print(f"Downloaded articles from general domains to {len(saved_files)} files")
        
    if not any([args.category, args.titles, args.popular, args.categories_file, 
                args.discover_categories, args.general_domains]):
        parser.error("One of --category, --titles, --popular, --categories-file, --discover-categories, or --general-domains must be specified")
    
    if args.category:
        # Use the domain-based download for categories that match our domain map
        saved_file = loader.download_articles_from_domain(
            args.category, 
            max_depth=args.max_depth,
            max_pages_per_category=args.max_pages,
            output_file=args.output_file
        )
        if saved_file:
            print(f"Downloaded articles from domain '{args.category}' to {saved_file}")
        else:
            # Fallback to single category download if not a known domain
            saved_file = loader.download_articles_from_category(
                args.category, 
                max_depth=args.max_depth,
                max_pages=args.max_pages,
                output_file=args.output_file
            )
            if saved_file:
                print(f"Downloaded articles from category '{args.category}' to {saved_file}")
    
    if args.titles:
        # Modify the download_articles_from_titles method to accept output_file parameter
        # This is not implemented in this edit
        saved_file = loader.download_articles_from_titles(args.titles)
        if saved_file:
            print(f"Downloaded articles from specified titles to {saved_file}")
    
    if args.categories_file:
        with open(args.categories_file, 'r', encoding='utf-8') as f:
            categories = [line.strip() for line in f if line.strip()]
        
        saved_files = loader.download_articles_from_multiple_categories(
            categories,
            max_depth=args.max_depth,
            max_pages_per_category=args.max_pages
        )
        print(f"Downloaded articles from categories in {args.categories_file} to {len(saved_files)} files")

if __name__ == "__main__":
    main()