# Japanese Kana-Kanji Conversion Pipeline

This pipeline automates the entire process of downloading Japanese Wikipedia articles, converting them from kanji to kana, filtering the data, creating datasets, and training models for kana-to-kanji conversion.

## Overview

The pipeline consists of the following steps:

1. **Download Wikipedia Articles**: Downloads articles from various categories in Japanese Wikipedia.
2. **Process Text**: Converts kanji text to kana using MeCab.
3. **Filter Data**: Cleans and filters the kana and kanji data.
4. **Format Dataset**: Creates Hugging Face datasets from the cleaned data.
5. **Train Model**: Fine-tunes a transformer model for kana-to-kanji conversion.

## Requirements

- Python 3.7+
- MeCab with a Japanese dictionary (optional but recommended)
- PyTorch (for GPU acceleration)
- Required Python packages (see `requirements.txt`)

## Usage

### Basic Usage

Run the complete pipeline with default settings:

```bash
cd transformer_ime
python run_pipeline.py
```

This will:
- Download Wikipedia articles from general domains (up to 200 pages per category)
- Process each category through all steps
- Train a model for each category

### Command-Line Options

#### Directory Settings

- `--data_dir`: Directory to store data files (default: "data/wiki")
- `--models_dir`: Directory to store trained models (default: "models")
- `--log_dir`: Directory to store log files (default: "logs")

#### Download Settings

- `--max_pages`: Maximum number of pages to download per category (default: 200)
- `--max_depth`: Maximum depth for category recursion (default: 1)
- `--skip_download`: Skip the download step (use existing files)
- `--categories`: Specific categories to process (skips download step)

#### Processing Settings

- `--mecab_path`: Path to MeCab dictionary (e.g., "/opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd")

#### Dataset Settings

- `--train_ratio`: Ratio of data to use for training (default: 0.8)
- `--test_ratio`: Ratio of data to use for testing (default: 0.1)
- `--validation_ratio`: Ratio of data to use for validation (default: 0.1)

#### Training Settings

- `--train`: Train models after preprocessing (default: True)
- `--no_train`: Skip model training
- `--model_name`: Base model to fine-tune (default: "ryos17/mt5_small_all")
- `--batch_size`: Per device batch size for training
- `--num_epochs`: Number of training epochs
- `--max_length`: Maximum sequence length

#### GPU Settings

- `--multi_gpu`: Enable multi-GPU training using distributed data parallel
- `--gpu_ids`: Specific GPU IDs to use (e.g., `--gpu_ids 0 2 3` to use GPUs 0, 2, and 3)

#### Advanced Training Settings

- `--learning_rate`: Learning rate for training
- `--weight_decay`: Weight decay for training
- `--warmup_steps`: Number of warmup steps
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--fp16`: Enable FP16 (mixed precision) training
- `--bf16`: Enable BF16 training (faster on newer GPUs)

### Examples

#### Process Only Specific Categories

```bash
python run_pipeline.py --categories Technology Science Art
```

#### Skip Download and Use Existing Files

```bash
python run_pipeline.py --skip_download
```

#### Download Only (No Training)

```bash
python run_pipeline.py --no_train
```

#### Use Custom MeCab Dictionary

```bash
python run_pipeline.py --mecab_path /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd
```

#### Train with Custom Parameters

```bash
python run_pipeline.py --model_name "google/mt5-small" --batch_size 32 --num_epochs 5
```

#### Multi-GPU Training

```bash
python run_pipeline.py --multi_gpu
```

#### Multi-GPU Training with Specific GPUs

```bash
python run_pipeline.py --multi_gpu --gpu_ids 0 2
```

#### Advanced Training Configuration

```bash
python run_pipeline.py --multi_gpu --batch_size 16 --learning_rate 3e-5 --warmup_steps 500 --fp16
```

## Output Structure

The pipeline creates the following directory structure:

```
data/
└── wiki/
    ├── Category1.kanji                 # Original kanji text
    ├── Category1.kana                  # Converted kana text
    ├── clean_Category1.kanji           # Filtered kanji text
    ├── clean_Category1.kana            # Filtered kana text
    └── formatted_Category1/            # HuggingFace dataset
        ├── train/                      # Training split (80% by default)
        ├── test/                       # Testing split (10% by default)
        └── validation/                 # Validation split (10% by default)

models/
└── category1/                          # Trained model for Category1
    ├── checkpoint-xxx/
    └── ...

logs/
└── pipeline-YYYYMMDD-HHMMSS.log        # Log file
```

## Multi-GPU Training

The pipeline supports multi-GPU training to accelerate the model training process. When using multi-GPU training:

1. The script automatically detects available GPUs on your system
2. It sets up the necessary environment variables for distributed training
3. The training process uses PyTorch's Distributed Data Parallel (DDP) to parallelize training across GPUs

### Requirements for Multi-GPU Training

- Multiple CUDA-capable GPUs
- PyTorch installed with CUDA support
- Sufficient system memory

### Tips for Multi-GPU Training

- Start with a smaller batch size per GPU and increase if memory allows
- Use `--fp16` or `--bf16` to reduce memory usage and increase training speed
- For very large models, consider using gradient accumulation with `--gradient_accumulation_steps`
- Monitor GPU usage with tools like `nvidia-smi` during training

## Troubleshooting

### Common Issues

1. **MeCab Not Found**: Ensure MeCab is installed and the path is correct.
2. **Memory Issues**: Reduce batch size or max_pages if you encounter memory problems.
3. **Failed Downloads**: Check your internet connection and try again.
4. **GPU Out of Memory**: Reduce batch size, use gradient accumulation, or enable mixed precision training.
5. **NCCL Errors**: These are related to GPU communication. Try setting `NCCL_DEBUG=INFO` environment variable for more information.

### Resuming Failed Runs

If some categories fail, you can resume processing just those categories:

```bash
python run_pipeline.py --categories FailedCategory1 FailedCategory2 --skip_download
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 