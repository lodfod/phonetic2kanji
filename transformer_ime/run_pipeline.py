#!/usr/bin/env python3
import os
import argparse
import subprocess
import logging
from pathlib import Path
from tqdm import tqdm
import time
import sys

def setup_logging(log_file=None):
    """Set up logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_command(cmd, logger):
    """Run a shell command and log the output"""
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Stream and log output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(line.strip())
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Command failed with return code {process.returncode}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return False

def download_wiki_articles(args, logger):
    """Download Wikipedia articles by category"""
    logger.info("=== STEP 1: Downloading Wikipedia articles ===")
    
    cmd = [
        "python", "preprocess/download_wiki.py",
        "--general-domains",
        "--max-pages", str(args.max_pages),
        "--output-dir", args.data_dir
    ]
    
    if args.max_depth:
        cmd.extend(["--max-depth", str(args.max_depth)])
    
    success = run_command(cmd, logger)
    
    if success:
        # Get list of downloaded kanji files
        kanji_files = list(Path(args.data_dir).glob("*.kanji"))
        logger.info(f"Downloaded {len(kanji_files)} category files")
        return [f.stem for f in kanji_files]  # Return category names
    else:
        logger.error("Failed to download Wikipedia articles")
        return []

def process_text(category, args, logger):
    """Process text from kanji to kana for a specific category"""
    logger.info(f"=== STEP 2: Processing text for category: {category} ===")
    
    kanji_file = os.path.join(args.data_dir, f"{category}.kanji")
    kana_file = os.path.join(args.data_dir, f"{category}.kana")
    
    cmd = [
        "python", "preprocess/text_processor.py",
        "--input", kanji_file,
        "--output", kana_file
    ]
    
    if args.mecab_path:
        cmd.extend(["--mecab_path", args.mecab_path])
    
    return run_command(cmd, logger)

def filter_data(category, args, logger):
    """Filter and clean data for a specific category"""
    logger.info(f"=== STEP 3: Filtering data for category: {category} ===")
    
    kana_file = os.path.join(args.data_dir, f"{category}.kana")
    kanji_file = os.path.join(args.data_dir, f"{category}.kanji")
    clean_kana_file = os.path.join(args.data_dir, f"clean_{category}.kana")
    clean_kanji_file = os.path.join(args.data_dir, f"clean_{category}.kanji")
    
    cmd = [
        "python", "preprocess/filter.py",
        "--kana", kana_file,
        "--kanji", kanji_file,
        "--clean_kana", clean_kana_file,
        "--clean_kanji", clean_kanji_file
    ]
    
    return run_command(cmd, logger)

def format_dataset(category, args, logger):
    """Format data into HuggingFace dataset for a specific category"""
    logger.info(f"=== STEP 4: Formatting dataset for category: {category} ===")
    
    clean_kana_file = os.path.join(args.data_dir, f"clean_{category}.kana")
    clean_kanji_file = os.path.join(args.data_dir, f"clean_{category}.kanji")
    output_dir = os.path.join(args.data_dir, f"formatted_{category}")
    
    cmd = [
        "python", "preprocess/dataset_formatter.py",
        "--kana_file", clean_kana_file,
        "--kanji_file", clean_kanji_file,
        "--output_dir", output_dir,
        "--train_ratio", str(args.train_ratio)
    ]
    
    return run_command(cmd, logger)

def train_model(category, args, logger):
    """Train model for a specific category"""
    logger.info(f"=== STEP 5: Training model for category: {category} ===")
    
    dataset_dir = os.path.join(args.data_dir, f"formatted_{category}")
    output_dir = os.path.join(args.models_dir, f"{category.lower()}")
    
    cmd = [
        "python", "src/train_reazon.py",
        "--dataset_dir", dataset_dir,
        "--output_dir", output_dir,
        "--model_name", args.model_name
    ]
    
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    
    if args.num_epochs:
        cmd.extend(["--num_epochs", str(args.num_epochs)])
    
    if args.max_length:
        cmd.extend(["--max_length", str(args.max_length)])
    
    if args.multi_gpu:
        cmd.append("--multi_gpu")
    
    return run_command(cmd, logger)

def process_category(category, args, logger):
    """Process a single category through the entire pipeline"""
    logger.info(f"\n{'='*50}\nProcessing category: {category}\n{'='*50}")
    
    # Step 2: Process text (kanji to kana)
    if not process_text(category, args, logger):
        logger.error(f"Failed to process text for category: {category}")
        return False
    
    # Step 3: Filter data
    if not filter_data(category, args, logger):
        logger.error(f"Failed to filter data for category: {category}")
        return False
    
    # Step 4: Format dataset
    if not format_dataset(category, args, logger):
        logger.error(f"Failed to format dataset for category: {category}")
        return False
    
    # Step 5: Train model (if enabled)
    if args.train:
        if not train_model(category, args, logger):
            logger.error(f"Failed to train model for category: {category}")
            return False
    
    logger.info(f"Successfully processed category: {category}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run the complete kana-kanji pipeline across all categories")
    
    # Directory settings
    parser.add_argument("--data_dir", default="data/wiki", help="Directory to store data files")
    parser.add_argument("--models_dir", default="models", help="Directory to store trained models")
    parser.add_argument("--log_dir", default="logs", help="Directory to store log files")
    
    # Download settings
    parser.add_argument("--max_pages", type=int, default=200, help="Maximum number of pages to download per category")
    parser.add_argument("--max_depth", type=int, default=1, help="Maximum depth for category recursion")
    parser.add_argument("--skip_download", action="store_true", help="Skip the download step (use existing files)")
    parser.add_argument("--categories", nargs="+", help="Specific categories to process (skip download step)")
    
    # Processing settings
    parser.add_argument("--mecab_path", help="Path to MeCab dictionary")
    
    # Dataset settings
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training")
    
    # Training settings
    parser.add_argument("--train", action="store_true", default=True, help="Train models after preprocessing")
    parser.add_argument("--no_train", action="store_false", dest="train", help="Skip model training")
    parser.add_argument("--model_name", default="ryos17/mt5_small_all", help="Base model to fine-tune")
    parser.add_argument("--batch_size", type=int, help="Per device batch size for training")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, help="Maximum sequence length")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU training")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up logging
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.log_dir, f"pipeline-{timestamp}.log")
    logger = setup_logging(log_file)
    
    logger.info(f"Starting pipeline with arguments: {args}")
    
    # Get categories to process
    categories = []
    if args.categories:
        categories = args.categories
        logger.info(f"Using specified categories: {categories}")
    elif not args.skip_download:
        # Step 1: Download Wikipedia articles
        categories = download_wiki_articles(args, logger)
        if not categories:
            logger.error("No categories downloaded or found. Exiting.")
            sys.exit(1)
    else:
        # Find existing kanji files if download is skipped
        kanji_files = list(Path(args.data_dir).glob("*.kanji"))
        categories = [f.stem for f in kanji_files]
        logger.info(f"Found {len(categories)} existing category files")
    
    # Process each category
    successful_categories = []
    failed_categories = []
    
    for category in tqdm(categories, desc="Processing categories"):
        if process_category(category, args, logger):
            successful_categories.append(category)
        else:
            failed_categories.append(category)
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info(f"Pipeline completed. Processed {len(categories)} categories.")
    logger.info(f"Successful: {len(successful_categories)}")
    logger.info(f"Failed: {len(failed_categories)}")
    
    if failed_categories:
        logger.info(f"Failed categories: {failed_categories}")
    
    logger.info(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    main() 