from datasets import load_dataset
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Process ReazonSpeech dataset and save transcriptions')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path')
    args = parser.parse_args()
    
    # Load reazonspeech corpus dataset
    ds = load_dataset("reazon-research/reazonspeech", "medium",
                     trust_remote_code=True,
                     num_proc=8)
    
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