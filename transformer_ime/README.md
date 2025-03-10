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
python preprocess/filter.py --kana data/reazon_tiny/data.kana --kanji data/reazon_tiny/data.kanji \
--clean_kana data/reazon_tiny/clean_data.kana --clean_kanji data/reazon_tiny/clean_data.kanji
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
```
Recommended to use ZeRO optimization level 0 
```bash
accelerate launch --main_process_port 29999 src/train_reazon.py --multi_gpu --dataset_dir data/reazon_tiny/formatted --output_dir models/my_test
```
Set your main process port to your desire (just make sure it is not used)

Fine-tuned model is now saved under `models/my_test`.

### 7. Upload Model to Hugging Face
```bash
python utils/upload_model.py --model_path models/my_test/final_model --hf_repo_id ryos17/my_test
```

## Evaluation

### For single line test 
```bash
python eval/inference.py \
  --model_name models/my_test/final_model \
  --input "ニッポンノアニメワコクサイテキニユウメイデス"
```
Note model_name can be hugging face directory or local model path

### For benchmarking mode

#### 1. Load evaluation dataset
```bash
python eval/dataloader.py --dataset jsut_basic5000 --output data/jsut_basic/data.kanji
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
python eval/benchmark.py --kana data/jsut_basic/clean_data.kana --kanji data/jsut_basic/clean_data.kanji --model_name ryos17/mt5_base_medium
```
Add --debug arguement to see sample preditions and references

#### Sample output (mt5_base_medium evaluated on jsut_basic)
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