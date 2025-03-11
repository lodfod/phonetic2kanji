#!/usr/bin/env python3
import os
import argparse
import glob
from pathlib import Path
import pandas as pd
from tabulate import tabulate
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def count_lines(file_path):
    """Count the number of lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception as e:
        return f"Error: {str(e)}"

def get_file_size(file_path):
    """Get the file size in a human-readable format."""
    try:
        size_bytes = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0 or unit == 'GB':
                break
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} {unit}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_file_size_bytes(file_path):
    """Get the file size in bytes for calculations."""
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        return 0

def analyze_directory(data_dir, output_format='table', output_file=None, plot=False):
    """
    Analyze the data directory structure and extract information about kana/kanji files.
    
    Args:
        data_dir: Path to the data directory
        output_format: Format for output ('table', 'json', or 'csv')
        output_file: Path to save the output (if None, print to console)
        plot: Whether to generate plots
    """
    data_dir = Path(data_dir)
    
    # Check if directory exists
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Error: Directory '{data_dir}' does not exist or is not a directory.")
        return
    
    # Initialize data structure to store results
    results = []
    
    # Find all kanji files (both original and cleaned)
    kanji_files = list(data_dir.glob("**/*.kanji"))
    
    # Process each kanji file and find corresponding kana file
    for kanji_file in kanji_files:
        kanji_path = str(kanji_file)
        kana_path = kanji_path.replace('.kanji', '.kana')
        
        # Skip if kana file doesn't exist
        if not os.path.exists(kana_path):
            continue
        
        # Get parent directory name
        parent_dir = kanji_file.parent.name
        
        # Get file basename
        file_basename = kanji_file.stem
        
        # Count lines
        kanji_lines = count_lines(kanji_path)
        kana_lines = count_lines(kana_path)
        
        # Get file sizes
        kanji_size = get_file_size(kanji_path)
        kana_size = get_file_size(kana_path)
        
        # Get raw file sizes for calculations
        kanji_size_bytes = get_file_size_bytes(kanji_path)
        kana_size_bytes = get_file_size_bytes(kana_path)
        
        # Calculate size ratio if possible
        try:
            size_ratio = kana_size_bytes / kanji_size_bytes if kanji_size_bytes > 0 else "N/A"
            if isinstance(size_ratio, float):
                size_ratio = f"{size_ratio:.2f}"
        except:
            size_ratio = "N/A"
        
        # Check if this is a clean file
        is_clean = "clean_" in file_basename
        
        # Check if there's a formatted dataset
        formatted_dir = data_dir / f"formatted_{file_basename.replace('clean_', '')}"
        has_formatted = formatted_dir.exists() and formatted_dir.is_dir()
        
        # Add to results
        results.append({
            "Category": file_basename.replace("clean_", ""),
            "Directory": parent_dir,
            "File Type": "Clean" if is_clean else "Original",
            "Kanji File": kanji_file.name,
            "Kana File": os.path.basename(kana_path),
            "Kanji Lines": kanji_lines,
            "Kana Lines": kana_lines,
            "Kanji Size": kanji_size,
            "Kana Size": kana_size,
            "Size Ratio (Kana/Kanji)": size_ratio,
            "Has Formatted Dataset": "Yes" if has_formatted else "No"
        })
    
    # Check for formatted datasets
    formatted_dirs = list(data_dir.glob("formatted_*"))
    formatted_data = []
    
    for formatted_dir in formatted_dirs:
        if not formatted_dir.is_dir():
            continue
            
        category = formatted_dir.name.replace("formatted_", "")
        
        # Check for train and test splits
        train_dir = formatted_dir / "train"
        test_dir = formatted_dir / "test"
        
        has_train = train_dir.exists() and train_dir.is_dir()
        has_test = test_dir.exists() and test_dir.is_dir()
        
        # Get dataset info if possible
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(str(formatted_dir))
            train_examples = len(dataset["train"]) if "train" in dataset else 0
            test_examples = len(dataset["test"]) if "test" in dataset else 0
        except:
            train_examples = "Unknown"
            test_examples = "Unknown"
        
        formatted_data.append({
            "Category": category,
            "Directory": formatted_dir.parent.name,
            "Dataset Path": str(formatted_dir),
            "Has Train Split": "Yes" if has_train else "No",
            "Has Test Split": "Yes" if has_test else "No",
            "Train Examples": train_examples,
            "Test Examples": test_examples
        })
    
    # If no results found
    if not results and not formatted_data:
        print(f"No kana/kanji files found in '{data_dir}'.")
        return
    
    # Convert to DataFrame for easier manipulation
    df_files = pd.DataFrame(results) if results else None
    df_datasets = pd.DataFrame(formatted_data) if formatted_data else None
    
    # Generate output
    if output_format == 'json':
        output = {
            "file_data": results if results else [],
            "dataset_data": formatted_data if formatted_data else []
        }
        output_str = json.dumps(output, indent=2)
    elif output_format == 'csv':
        output_parts = []
        if df_files is not None:
            output_parts.append("# File Data\n" + df_files.to_csv(index=False))
        if df_datasets is not None:
            output_parts.append("# Dataset Data\n" + df_datasets.to_csv(index=False))
        output_str = "\n\n".join(output_parts)
    else:  # table format
        output_parts = []
        if df_files is not None:
            output_parts.append("File Data:\n" + tabulate(df_files, headers='keys', tablefmt='grid', showindex=False))
        if df_datasets is not None:
            output_parts.append("\nDataset Data:\n" + tabulate(df_datasets, headers='keys', tablefmt='grid', showindex=False))
        output_str = "\n\n".join(output_parts)
    
    # Output results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print(f"Results saved to {output_file}")
    else:
        print(output_str)
    
    # Generate plots if requested
    if plot and df_files is not None:
        generate_plots(df_files, data_dir)

def generate_plots(df, data_dir):
    """Generate plots for data visualization."""
    try:
        # Create a plots directory
        plots_dir = Path(data_dir) / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Filter to get only clean files for better comparison
        clean_df = df[df["File Type"] == "Clean"].copy()
        
        if len(clean_df) > 0:
            # Convert line counts to numeric
            clean_df["Kanji Lines"] = pd.to_numeric(clean_df["Kanji Lines"], errors='coerce')
            clean_df["Kana Lines"] = pd.to_numeric(clean_df["Kana Lines"], errors='coerce')
            
            # Plot line counts by category
            plt.figure(figsize=(12, 6))
            clean_df.plot(x="Category", y=["Kanji Lines", "Kana Lines"], kind="bar", figsize=(12, 6))
            plt.title("Number of Lines by Category")
            plt.ylabel("Number of Lines")
            plt.tight_layout()
            plt.savefig(plots_dir / "line_counts.png")
            
            # Plot size ratio
            clean_df["Size Ratio (Kana/Kanji)"] = pd.to_numeric(clean_df["Size Ratio (Kana/Kanji)"], errors='coerce')
            plt.figure(figsize=(12, 6))
            clean_df.plot(x="Category", y="Size Ratio (Kana/Kanji)", kind="bar", figsize=(12, 6))
            plt.title("Kana/Kanji Size Ratio by Category")
            plt.ylabel("Size Ratio")
            plt.tight_layout()
            plt.savefig(plots_dir / "size_ratio.png")
            
            print(f"Plots saved to {plots_dir}")
    except Exception as e:
        print(f"Error generating plots: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze kana/kanji data files in a directory")
    parser.add_argument("--data_dir", required=True, help="Path to the data directory")
    parser.add_argument("--output", choices=["table", "json", "csv"], default="table", 
                        help="Output format (default: table)")
    parser.add_argument("--output_file", help="Path to save the output (if not specified, print to console)")
    parser.add_argument("--plot", action="store_true", help="Generate plots for data visualization")
    
    args = parser.parse_args()
    
    analyze_directory(args.data_dir, args.output, args.output_file, args.plot)

if __name__ == "__main__":
    main() 