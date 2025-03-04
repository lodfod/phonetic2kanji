# Transformer IME Usage (For Ryota)

## Setup

```bash
cd transformer_ime
```

### Requirements (WITHIN THIS DIRECTORY, SLIGHTLY DIFFERENT FROM PARENT PROJECT)

```bash
pip install -r requirements.txt
```

## Training Pipeline

### 1. Create a sample dataset

```bash
python preprocess/download_reazon.py --output data/reazon/reazon_data.json --size tiny --split train --max_samples 1000
```

### 2. Extract kanji text from the sample dataset

```bash
python preprocess/reazon_dataloader.py --input data/reazon/reazon_data.json --output data/reazon/data.kanji --shuffle
```

### 3. Convert kanji to kana

```bash
python preprocess/text_processor.py --input data/reazon/data.kanji --output data/reazon/data.kana --mecab_path /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd (REPLACE WITH YOUR PATH TO MECAB DICT)
```

### 4. Format for Hugging Face

```bash
python preprocess/dataset_formatter.py --kana_file data/reazon/data.kana --kanji_file data/reazon/data.kanji --output_dir data/reazon/formatted --train_ratio 0.8
```

### 5. Validate the formatted dataset (optional)

```bash
python utils/data_validator.py --dataset_path data/reazon/formatted
```

### 6. Fine-tune Model

```bash
python src/train_reazon.py --dataset_dir data/reazon/formatted --output_dir models/base_model --num_epochs 3
```

Fine-tuned model is now saved under `models/base_model`.

## Perform Inference

### Single test

```bash
python src/inference.py \
  --model_dir models/base_model/final_model \
  --input "ニッポンノアニメハコクサイテキニユウメイデス。"
```

### Evaluate Model (UNTESTED)

```bash
python src/evaluate_model.py \
  --model_dir models/base_model/final_model \
  --data_dir data/reazon \
  --output_file base_model_evaluation.txt
```