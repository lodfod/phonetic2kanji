#!/usr/bin/env python3
import os
import argparse
import subprocess
import logging
from pathlib import Path
from tqdm import tqdm
import time
import sys
import shutil

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

def detect_gpus():
    """Detect available GPUs and their properties"""
    try:
        # Try to import torch
        import torch
        
        if not torch.cuda.is_available():
            return {"available": False, "count": 0, "names": [], "cuda_version": None}
        
        count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(count)]
        cuda_version = torch.version.cuda
        
        return {
            "available": True,
            "count": count,
            "names": names,
            "cuda_version": cuda_version
        }
    except ImportError:
        # If torch is not available, try using nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            gpu_names = [name.strip() for name in result.stdout.strip().split('\n') if name.strip()]
            
            return {
                "available": len(gpu_names) > 0,
                "count": len(gpu_names),
                "names": gpu_names,
                "cuda_version": "Unknown"
            }
        except (subprocess.SubprocessError, FileNotFoundError):
            # If nvidia-smi fails or is not found
            return {"available": False, "count": 0, "names": [], "cuda_version": None}

def setup_gpu_environment(multi_gpu, gpu_ids=None, logger=None):
    """Set up environment variables for GPU training"""
    if logger:
        logger.info("Setting up GPU environment...")
    
    # Detect available GPUs
    gpu_info = detect_gpus()
    
    if logger:
        if gpu_info["available"]:
            logger.info(f"Found {gpu_info['count']} GPUs: {', '.join(gpu_info['names'])}")
            logger.info(f"CUDA version: {gpu_info['cuda_version']}")
        else:
            logger.warning("No GPUs detected. Training will use CPU only.")
    
    # Set environment variables
    if multi_gpu and gpu_info["available"] and gpu_info["count"] > 1:
        # For multi-GPU training
        if logger:
            logger.info("Setting up environment for multi-GPU training")
        
        # If specific GPU IDs are provided, use them
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            if logger:
                logger.info(f"Using GPUs with IDs: {gpu_ids}")
        
        # Set distributed training environment variables
        os.environ["NCCL_TIMEOUT"] = "3600"
        os.environ["NCCL_IB_TIMEOUT"] = "120"
        os.environ["NCCL_SOCKET_TIMEOUT"] = "120"
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1000"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        return True
    elif gpu_info["available"]:
        # For single-GPU training
        if logger:
            logger.info("Setting up environment for single-GPU training")
        
        # If specific GPU ID is provided, use it
        if gpu_ids and len(gpu_ids) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
            if logger:
                logger.info(f"Using GPU with ID: {gpu_ids[0]}")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU by default
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Clear any existing distributed training settings
        for var in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
            if var in os.environ:
                del os.environ[var]
        
        return True
    else:
        # No GPUs available
        if logger:
            logger.warning("No GPUs available. Using CPU for training.")
        
        # Clear any GPU-related environment variables
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        
        return False

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
    
    # Add test and validation ratios if provided
    if hasattr(args, 'test_ratio'):
        cmd.extend(["--test_ratio", str(args.test_ratio)])
    
    if hasattr(args, 'validation_ratio'):
        cmd.extend(["--validation_ratio", str(args.validation_ratio)])
    
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
    
    # Add additional training arguments if provided
    if args.learning_rate:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    
    if args.weight_decay:
        cmd.extend(["--weight_decay", str(args.weight_decay)])
    
    if args.warmup_steps:
        cmd.extend(["--warmup_steps", str(args.warmup_steps)])
    
    if args.gradient_accumulation_steps:
        cmd.extend(["--gradient_accumulation_steps", str(args.gradient_accumulation_steps)])
    
    if args.fp16:
        cmd.append("--fp16")
    
    if args.bf16:
        cmd.append("--bf16")
    
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
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training (default: 0.8)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of data to use for testing (default: 0.1)")
    parser.add_argument("--validation_ratio", type=float, default=0.1, help="Ratio of data to use for validation (default: 0.1)")
    
    # Training settings
    parser.add_argument("--train", action="store_true", default=True, help="Train models after preprocessing")
    parser.add_argument("--no_train", action="store_false", dest="train", help="Skip model training")
    parser.add_argument("--model_name", default="ryos17/mt5_small_all", help="Base model to fine-tune")
    parser.add_argument("--batch_size", type=int, help="Per device batch size for training")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, help="Maximum sequence length")
    
    # GPU settings
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU training")
    parser.add_argument("--gpu_ids", type=int, nargs="+", help="Specific GPU IDs to use (e.g., 0 1 2)")
    
    # Advanced training settings
    parser.add_argument("--learning_rate", type=float, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, help="Weight decay for training")
    parser.add_argument("--warmup_steps", type=int, help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 training")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 training")
    
    args = parser.parse_args()
    
    # Verify that dataset split ratios sum to 1.0
    total_ratio = args.train_ratio + args.test_ratio + args.validation_ratio
    if not 0.999 <= total_ratio <= 1.001:  # Allow for small floating-point errors
        logger.error(f"Train, test, and validation ratios must sum to 1.0, got {total_ratio}")
        sys.exit(1)
    
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up logging
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.log_dir, f"pipeline-{timestamp}.log")
    logger = setup_logging(log_file)
    
    logger.info(f"Starting pipeline with arguments: {args}")
    
    # Set up GPU environment
    gpu_available = setup_gpu_environment(args.multi_gpu, args.gpu_ids, logger)
    
    if args.multi_gpu and not gpu_available:
        logger.warning("Multi-GPU training requested but no GPUs detected. Falling back to CPU.")
    
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