from datasets import load_dataset
import argparse
from pathlib import Path
from huggingface_hub import login

def main():
    parser = argparse.ArgumentParser(description='Process ReazonSpeech dataset and save transcriptions')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path')
    parser.add_argument('--size', type=str, default='medium',
                       choices=['tiny', 'small', 'medium', 'large', 'all'],
                       help='Size of the dataset to load (default: medium)')
    parser.add_argument('--num_proc', type=int, default=8,
                       help='Number of processes to use for loading (default: 8)')
    args = parser.parse_args()
    
    # Load reazonspeech corpus dataset
    ds = load_dataset("reazon-research/reazonspeech", args.size,
                     trust_remote_code=True,
                     num_proc=args.num_proc)
    
    # Extract transcriptions and save directly to txt
    ds['train'].to_csv(
        args.output,
        index=False,
        header=False,
        columns=['transcription']
    )
    
    # Remove the last empty line more reliably
    with open(args.output, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines[:-1]))

if __name__ == '__main__':
    main()