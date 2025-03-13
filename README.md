# Japanese Kana-to-Kanji Conversion System

This repository contains an implementation of a phonetic-to-kanji conversion system for Japanese text, built using Hugging Face's Transformers library. The system leverages sequence-to-sequence models fine-tuned on extensive Japanese language datasets to accurately convert kana (phonetic characters) to appropriate kanji.

The implementation provides a complete pipeline from data preprocessing to model training and evaluation. Sample commands for running each component are included below, while detailed hyperparameters and training configurations used to reproduce the results in our paper can be found in the accompanying project report. Details of each command args are on its respective python file. 

## Project Structure

- `data/` - Contains original data and processed datasets
- `models/` - Contains trained model
- `preprocess/` - Scripts for data preprocessing
- `train_and_eval/` - Source code for model training and evaluation
- `utils/` - Other important utility functions for data handling and management

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

## General-Context Fine-Tuning Pipeline

### 1. Download and extract kanji text from Reazon dataset

```bash
python preprocess/download_reazon.py --output data/reazon_medium/data.kanji --size medium
```

### 2. Convert kanji to kana

```bash
python preprocess/text_processor.py --input data/reazon_medium/data.kanji --output data/reazon_medium/data.kana
```

### 3. Remove and filter unwanted data
```bash
python preprocess/filter.py --kana data/reazon_medium/data.kana --kanji data/reazon_medium/data.kanji \
--clean_kana data/reazon_medium/clean_data.kana --clean_kanji data/reazon_medium/clean_data.kanji
```

### 4. Format for Hugging Face

```bash
python preprocess/dataset_formatter.py --kana_file data/reazon_medium/clean_data.kana \
--kanji_file data/reazon_medium/clean_data.kanji --output_dir data/reazon_medium/formatted \
--train_ratio 0.8 --validation_ratio 0.2 --test_ratio 0
```

### 5. Fine-tune model

```bash
python train_and_eval/train.py --dataset_dir data/reazon_medium/formatted --model_name google/mt5-small \
--output_dir models/mt5_small_medium
```
Fine-tuned model is now saved under `models/mt5_small_medium`.

### For multi-GPU fine-tuning
#### Configure multi gpu training parameters
```bash
accelerate config
```
Recommended to use ZeRO optimization level 0 
#### Fine-tune model using multi-GPU 
```bash
accelerate launch --main_process_port 29999 src/train.py --multi_gpu --dataset_dir data/reazon_medium/formatted \
--model_name google/mt5-small --output_dir models/mt5_small_medium 
```
Set your main process port to your desire (just make sure it is not used)

## General-Context Evaluation

### For single line test 
```bash
python train_and_eval/inference.py \
  --model_name models/mt5_small_medium/final_model \
  --input "ニッポンノアニメワコクサイテキニユウメイデス"
```
Note: model_name can be hugging face directory or local model path

### For benchmarking model

#### 1. Load evaluation dataset
```bash
python preprocess/download_general_eval.py --dataset jsut_basic5000 --output data/jsut_basic/data.kanji
```
Choose between jsut_basic5000 or common_voice_8_0

#### 2. Convert Kanji to Kana
```bash
python preprocess/text_processor.py --input data/jsut_basic/data.kanji --output data/jsut_basic/data.kana
```

#### 3. Remove and filter unwanted data
```bash
python preprocess/filter.py --kana data/jsut_basic/data.kana --kanji data/jsut_basic/data.kanji \
--clean_kana data/jsut_basic/clean_data.kana --clean_kanji data/jsut_basic/clean_data.kanji
```

#### 4. Evaluate model
```bash
python train_and_eval/benchmark.py --kana data/jsut_basic/clean_data.kana \
--kanji data/jsut_basic/clean_data.kanji --model_name ryos17/mt5_small_medium
```
Note: model_name can be hugging face directory or local model path

Add --debug arguement to see sample preditions and references

#### Sample output: mt5_small_medium model evaluated on jsut_basic
```bash
================================================================================
Overall Results:
Precision: 63.5646
Recall: 63.4780
F-score: 63.3499
Character Error Rate (CER): 45.5615
Sentence Accuracy: 0.6463
================================================================================
```
## Domain-Specific Fine-Tuning Pipeline

### 1. Download and extract kanji text from domain-specific Wikipedia 

```bash
python preprocess/download_wiki.py --category technology --max-pages 200 --output-dir data/wiki_tech/data.kanji
```

### 2. Convert kanji to kana

```bash
python preprocess/text_processor.py --input data/wiki_tech/data.kanji --output data/wiki_tech/data.kana
```

### 3. Remove and filter unwanted data
```bash
python preprocess/filter.py --kana data/wiki_tech/data.kana --kanji data/wiki_tech/data.kanji \
--clean_kana data/wiki_tech/clean_data.kana --clean_kanji data/wiki_tech/clean_data.kanji
```

### 4. Format for Hugging Face

```bash
python preprocess/dataset_formatter.py --kana_file data/wiki_tech/clean_data.kana \
--kanji_file data/wiki_tech/clean_data.kanji --output_dir data/wiki_tech/formatted \
--train_ratio 0.8 --validation_ratio 0.1 --test_ratio 0.1
```

### 5. Fine-tune model

```bash
python train_and_eval/train.py --dataset_dir data/wiki_tech/formatted --model_name ryos17/mt5_base_all \
--output_dir models/wiki_technology_epoch_4 --num_epochs 4
```
For `--model_name`, choose genral-context fine-tuned model. Note: model_name can be hugging face directory or local model path.

Fine-tuned model is now saved under `models/wiki_technology_epoch_4`.

## Domain-Specific Evaluation

### For single line test 
```bash
python train_and_eval/inference.py \
  --model_name models/wiki_technology_epoch_4/final_model \
  --input "ニッポンノアニメワコクサイテキニユウメイデス"
```
Note: model_name can be hugging face directory or local model path

### For benchmarking model

#### 1. Load evaluation dataset
```bash
python preprocess/download_wiki_eval.py --dataset_path data/wiki_tech/formatted \
--kana_output data/wiki_tech/test.kana --kanji_output data/wiki_tech/test.kanji
```
#### 2. Evaluate model
```bash
python train_and_eval/benchmark.py --kana data/wiki_tech/test.kana \
--kanji data/wiki_tech/test.kanji --model_name models/wiki_technology_epoch_4/final_model
```
Note: model_name can be hugging face directory or local model path

Add --debug arguement to see sample preditions and references

#### Sample output: Evaluating models/wiki_technology_epoch_4/final_model/ on data/wiki_tech/test.kana and data/wiki_tech/test.kanji dataset
```bash
================================================================================
Overall Results:
Precision: 84.2160
Recall: 84.4955
F-score: 84.2909
Character Error Rate (CER): 18.2989
Sentence Accuracy: 13.5528
================================================================================
```

## Optional Utility Commands

### Upload Model to Hugging Face for Open Use
```bash
python utils/upload_model.py --model_path models/mt5_small_medium/final_model --hf_repo_id ryos17/mt5_small_medium
```

### Validate dataset
```bash
python utils/data_validator.py --dataset_path data/reazon_tiny/formatted
python utils/vocab.py --input data/reazon_tiny/clean_data.kana --output data/reazon_tiny/vocab.json
```

### Dictionary of unique characters in given data file
```bash
python utils/vocab.py --input data/reazon_tiny/clean_data.kana --output data/reazon_tiny/vocab.json
```

### Run benchmark on non neural model
```bash
python utils/non_neural_ime.py --kana_input data/common_voice/clean_data.kana --kanji_output data/common_voice/non_neural.kanji
python utils/non_neural_benchmark.py --prediction data/common_voice/non_neural.kanji --reference data/common_voice/clean_data.kanji
```