# Transformer-based IME for Kana to Kanji Conversion

This project implements a kana to kanji conversion system using Hugging Face's Transformers library. The system is based on a sequence-to-sequence model fine-tuned on Japanese text data.

## Project Structure

- `data/` - Contains vocabulary files and processed datasets
- `preprocess/` - Scripts for data preprocessing
- `src/` - Source code for model training and inference
- `eval/` - Evaluation scripts and metrics
- `utils/` - Utility functions for data handling and tokenization

## Setup

### Requirements 

```bash
pip install -r requirements.txt
```

### MeCab Installation

```bash
apt update
apt install -y mecab libmecab-dev mecab-ipadic mecab-ipadic-utf8 swig sudo git curl xz-utils
git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
./mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -p "$(pwd)/mecab-ipadic-neologd/lib"
```

## Data Preparation

### Reazon Dataset

1. Create kanji data:

```bash
python preprocess/reazon_dataloader.py --output data/reazon/data.kanji
```

2. Create kana data:

```bash
python preprocess/text_processor.py --input data/reazon/data.kanji --output data/reazon/data.kana --mecab_path /path/to/mecab-ipadic-neologd/lib
```

3. Format data for Hugging Face:

```bash
python preprocess/dataset_formatter.py --kana_file data/reazon/data.kana --kanji_file data/reazon/data.kanji --output_dir data/reazon/formatted
```

### Wikipedia Dataset

4. Process Wikipedia data:

```bash
python preprocess/text_processor.py --input data/wiki/wiki_dump.txt --output data/wiki/wiki.kana --mecab_path /path/to/mecab-ipadic-neologd/lib --wiki --max_sentences 100000
```

5. Format data for Hugging Face:
   
```bash
python preprocess/dataset_formatter.py --kana_file data/wiki/wiki.kana --kanji_file data/wiki/wiki.kanji --output_dir data/wiki/formatted
```

## Model Training

### Base Model Training (Reazon Dataset)

```bash
python src/train_reazon.py --dataset_dir data/reazon/formatted --output_dir models/base_model --num_epochs 5
```

### Domain-Specific Fine-tuning (Wikipedia)

```bash
python src/train_wiki.py --base_model_dir models/base_model/final_model --dataset_dir data/wiki/formatted --output_dir models/wiki_finetuned --num_epochs 3
```

## Inference

### Convert Single Input

```bash
python src/inference.py --model_dir models/wiki_finetuned/final_model --input "こんにちは"
```

### Batch Processing

```bash
python src/inference.py --model_dir models/wiki_finetuned/final_model --input data/test/test.kana --batch
```

## Evaluation

```bash
python eval/evaluator.py --model_dir models/wiki_finetuned/final_model --test_kana data/test/test.kana --test_kanji data/test/test.kanji --output_dir eval_results
```

## Training Pipeline Overview

1. **Data Preprocessing**: Convert raw text to kana-kanji pairs and format for transformer training
2. **Base Model Training**: Train on general-purpose Reazon dataset
3. **Domain-Specific Fine-tuning**: Further fine-tune on Wikipedia data
4. **Evaluation**: Measure performance using character error rate (CER)

## Model Details

This implementation uses the T5 model architecture which has shown strong performance on sequence-to-sequence tasks. The model treats kana to kanji conversion as a translation task, where the source language is kana and the target language is kanji.

Key advantages of this approach:

- Leverages pre-trained language models
- Handles context-dependent conversion effectively
- Can be fine-tuned for specific domains

