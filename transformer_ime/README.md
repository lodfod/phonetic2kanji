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

### 1. Create a sample dataset and extract kanji text from the sample dataset

```bash
python preprocess/download_reazon.py --output data/reazon/reazon_data.json --size tiny --split train --max_samples 1000
python preprocess/reazon_dataloader.py --input data/reazon/reazon_data.json --output data/reazon/data.kanji --shuffle
```

#### Alternatively, download and extract kanji text in one script

```bash
python preprocess/updated_reazon_dataloader.py --output data/reazon_tiny/data.kanji --size tiny
```

### 2. Convert kanji to kana

```bash
python preprocess/text_processor.py --input data/reazon_tiny/data.kanji --output data/reazon_tiny/data.kana
```

### 3. Remove and filter unwanted data
```bash
python preprocess/filter.py --kana data/reazon_tiny/data.kana --kanji data/reazon_tiny/data.kanji --clean_kana data/reazon_tiny/clean_data.kana --clean_kanji data/reazon_tiny/clean_data.kanji
```

### 4. Format for Hugging Face

```bash
python preprocess/dataset_formatter.py --kana_file data/reazon_tiny/clean_data.kana --kanji_file data/reazon_tiny/clean_data.kanji --output_dir data/reazon_tiny/formatted --train_ratio 0.8
```

### 5. Validate the formatted dataset (optional) and vocabulary dictionary 

```bash
python utils/data_validator.py --dataset_path data/reazon_tiny/formatted
python utils/vocab.py --input data/reazon_tiny/clean_data.kana --output data/reazon_tiny/vocab.json
```

### 6. Fine-tune Model

```bash
python src/train_reazon.py --dataset_dir data/reazon_tiny/formatted --output_dir models/my_test
```

#### For multi-GPU Training
```bash
accelerate config
accelerate launch --main_process_port 0 src/train_reazon.py --multi_gpu --dataset_dir data/reazon_tiny/formatted --output_dir models/my_test
```

Fine-tuned model is now saved under `models/my_test`.

### 7. Upload Model to Hugging Face
```bash
python utils/upload_model.py --model_path models/my_test/final_model --hf_repo_id ryos17/my_test
```

## Perform Inference

### Single test

```bash
python src/inference.py \
  --model_dir models/my_test/final_model \
  --input "ニッポンノアニメワコクサイテキニユウメイデス"
```

### Evaluate Model (UNTESTED)

```bash
python src/evaluate_model.py \
  --model_dir models/base_model/final_model \
  --data_dir data/reazon \
  --output_file base_model_evaluation.txt
```