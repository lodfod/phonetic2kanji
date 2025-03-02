import os
import wikipediaapi
import argparse
import logging
import time
import re
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikipediaDataLoader:
    def __init__(self, language='ja', output_dir='wiki_articles'):
        """
        Initialize the Wikipedia data loader.
        
        Args:
            language (str): Language code for Wikipedia (default: 'ja' for Japanese)
            output_dir (str): Directory to save the downloaded articles
        """
        self.language = language
        self.output_dir = output_dir
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='WikipediaDataLoader/1.0 (your-email@example.com)',
            language=language
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Download Japanese sentence tokenizer if needed
        try:
            nltk.download('punkt')
        except Exception as e:
            logger.warning(f"Error downloading NLTK punkt: {e}")
            logger.info("Will use simple regex-based sentence splitting as fallback")

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
            logger.error(f"Category '{category_name}' does not exist")
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
                logger.debug(f"Added page: {page.title}")
                
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
            logger.error(f"Page '{page_title}' does not exist")
            return None
        
        # Get the full text of the article
        text = page.text
        
        # Try to use NLTK's sentence tokenizer
        try:
            sentences = sent_tokenize(text, language='japanese')
        except LookupError:
            # Fallback to simple regex-based sentence splitting
            # Japanese sentences typically end with one of these characters
            logger.info("Using fallback sentence tokenizer")
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

    def save_article(self, page, category=None, filename=None):
        """
        Save an article to a file with one sentence per line.
        
        Args:
            page (WikipediaPage): Wikipedia page object
            category (str, optional): Category name for subfolder organization
            filename (str, optional): Custom filename. If None, use the page title.
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            # Create a safe filename from the page title
            filename = re.sub(r'[^\w\s-]', '', page.title).strip().replace(' ', '_')
        
        # Determine the output directory (create category subfolder if specified)
        output_dir = self.output_dir
        if category:
            # Create a safe folder name from the category
            safe_category = re.sub(r'[^\w\s-]', '', category).strip().replace(' ', '_')
            category_dir = os.path.join(self.output_dir, safe_category)
            os.makedirs(category_dir, exist_ok=True)
            output_dir = category_dir
        
        filepath = os.path.join(output_dir, f"{filename}.txt")
        
        # Get article text with one sentence per line
        article_text = self.download_article(page.title)
        
        if article_text:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(article_text)
            logger.info(f"Saved article: {page.title} to {filepath}")
            return filepath
        else:
            logger.warning(f"Failed to save article: {page.title}")
            return None

    def download_articles_from_category(self, category_name, max_depth=1, max_pages=100):
        """
        Download all articles from a category.
        
        Args:
            category_name (str): Name of the category
            max_depth (int): Maximum depth for category recursion
            max_pages (int): Maximum number of pages to retrieve
            
        Returns:
            list: List of paths to saved files
        """
        pages = self.get_category_members(category_name, max_depth, max_pages)
        
        saved_files = []
        for page in tqdm(pages, desc=f"Downloading articles from {category_name}"):
            # Pass the category name to save_article
            filepath = self.save_article(page, category=category_name)
            if filepath:
                saved_files.append(filepath)
            
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.5)
            
        return saved_files

    def download_articles_from_titles(self, titles):
        """
        Download articles from a list of titles.
        
        Args:
            titles (list): List of page titles
            
        Returns:
            list: List of paths to saved files
        """
        saved_files = []
        for title in tqdm(titles, desc="Downloading articles"):
            page = self.wiki.page(title)
            if page.exists():
                filepath = self.save_article(page)
                if filepath:
                    saved_files.append(filepath)
            else:
                logger.warning(f"Page '{title}' does not exist")
                
            # Add a small delay to avoid hitting rate limits
            time.sleep(0.5)
            
        return saved_files

    def download_articles_from_multiple_categories(self, category_names, max_depth=1, max_pages_per_category=100):
        """
        Download articles from multiple categories.
        
        Args:
            category_names (list): List of category names
            max_depth (int): Maximum depth for category recursion
            max_pages_per_category (int): Maximum number of pages to retrieve per category
            
        Returns:
            list: List of paths to saved files
        """
        all_saved_files = []
        
        for category_name in category_names:
            logger.info(f"Processing category: {category_name}")
            saved_files = self.download_articles_from_category(
                category_name, 
                max_depth=max_depth,
                max_pages=max_pages_per_category
            )
            all_saved_files.extend(saved_files)
            logger.info(f"Downloaded {len(saved_files)} articles from category '{category_name}'")
            
            # Add a delay between categories to avoid hitting rate limits
            time.sleep(2)
        
        return all_saved_files


    def find_general_domain_categories(self):
        """
        Returns a list of general domain categories in Japanese Wikipedia.
        
        Returns:
            list: List of category names covering diverse domains
        """
        # Map of general domains to their Japanese Wikipedia category names #TODO: Improve category search
        general_domains = {
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
        
        # Flatten the list of categories
        all_categories = []
        for domain, categories in general_domains.items():
            for category in categories:
                all_categories.append(category)
        
        logger.info(f"Using {len(all_categories)} general domain categories across {len(general_domains)} domains")
        return all_categories

    def download_general_domain_articles(self, max_depth=1, max_pages_per_category=20):
        """
        Download articles from general domains in Japanese.
        
        Args:
            max_depth (int): Maximum depth for category recursion
            max_pages_per_category (int): Maximum number of pages to retrieve per category
            
        Returns:
            list: List of paths to saved files
        """
        categories = self.find_general_domain_categories()
        return self.download_articles_from_multiple_categories(
            categories,
            max_depth=max_depth,
            max_pages_per_category=max_pages_per_category
        )

def main():
    parser = argparse.ArgumentParser(description='Download Wikipedia articles with one sentence per line')
    
    parser.add_argument('--category', type=str, help='Category to download articles from')
    parser.add_argument('--titles', type=str, nargs='+', help='Specific article titles to download')
    parser.add_argument('--output-dir', type=str, default='wiki_articles', help='Output directory')
    parser.add_argument('--language', type=str, default='ja', help='Wikipedia language code')
    parser.add_argument('--max-depth', type=int, default=1, help='Maximum category recursion depth')
    parser.add_argument('--max-pages', type=int, default=100, help='Maximum number of pages to download')
    parser.add_argument('--popular', action='store_true', help='Download articles from popular Japanese categories')
    parser.add_argument('--num-categories', type=int, default=10, help='Number of popular categories to use')
    parser.add_argument('--categories-file', type=str, help='File containing category names, one per line')
    parser.add_argument('--general-domains', action='store_true', 
                        help='Download articles from general domains (Physics, Medical, Tech, etc.) in Japanese')
    parser.add_argument('--discover-categories', action='store_true',
                        help='Discover and use popular categories automatically')
    
    args = parser.parse_args()

    loader = WikipediaDataLoader(language=args.language, output_dir=args.output_dir)
    if args.general_domains:
        saved_files = loader.download_general_domain_articles(
            max_depth=args.max_depth,
            max_pages_per_category=args.max_pages
        )
        logger.info(f"Downloaded {len(saved_files)} articles from general domains")
    if not any([args.category, args.titles, args.popular, args.categories_file, 
                args.discover_categories, args.general_domains]):
        parser.error("One of --category, --titles, --popular, --categories-file, --discover-categories, or --general-domains must be specified")
    
    
    if args.category:
        saved_files = loader.download_articles_from_category(
            args.category, 
            max_depth=args.max_depth,
            max_pages=args.max_pages
        )
        logger.info(f"Downloaded {len(saved_files)} articles from category '{args.category}'")
    
    if args.titles:
        saved_files = loader.download_articles_from_titles(args.titles)
        logger.info(f"Downloaded {len(saved_files)} articles from specified titles")
    
    if args.popular:
        saved_files = loader.download_popular_categories(
            num_categories=args.num_categories,
            max_depth=args.max_depth,
            max_pages_per_category=args.max_pages
        )
        logger.info(f"Downloaded {len(saved_files)} articles from popular categories")
    
    if args.categories_file:
        with open(args.categories_file, 'r', encoding='utf-8') as f:
            categories = [line.strip() for line in f if line.strip()]
        
        saved_files = loader.download_articles_from_multiple_categories(
            categories,
            max_depth=args.max_depth,
            max_pages_per_category=args.max_pages
        )
        logger.info(f"Downloaded {len(saved_files)} articles from categories in {args.categories_file}")

if __name__ == "__main__":
    main()
