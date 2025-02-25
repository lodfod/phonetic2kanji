# phonetic2kanji

## Setting up environment
### Install conda environment
```
cd abdp_ime
conda env create --name envname --file=env.yml
conda activate envname
```
### Install Mecab with extended dictionary
```
apt update
apt install -y ffmpeg mecab libmecab-dev mecab-ipadic mecab-ipadic-utf8 swig sudo git curl xz-utils
git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
./mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -p "$(pwd)/mecab-ipadic-neologd/lib"
```

## Creating Dataset

### Creating Reazon Dataset
```
python preprocess/reazon_dataloader.py --output 'data/data.kanji'
```
### Creating the .kana equivalent file
```
python preprocess/mecab_processor.py --input 'data/data.kanji' --output 'data/data.kana'
```

### Using dummy data
#### data/data.kana
```
こ の じ て ん で わ れ わ れ は か み の こ と ば 、
「 わ が お も い は な ん じ の お も い で は な い 。
```

#### data/data.kanji
```
この 時点 で われわれ は 神 の 言葉 、
「 我が 思い は 汝 の 思い で は ない 。
```

## Pipeline Steps

### 1. Preprocessing

#### Extract Alignment
```
python preprocess/mec2alignment.py data/data.kana data/data.kanji   
```

#### Create Tokens
```
python preprocess/tokenization.py --tokenizer_path data/vocabs/kana.json --train_files data/data.kana --files_to_conv data/data.kana  --vocab_size 500 --algorithm wordlevel
```

```
python preprocess/tokenization.py --tokenizer_path data/vocabs/train_kanji.json --train_files data/data.kanji --files_to_conv data/data.kanji --vocab_size 16000 --algorithm bpe 
```

### 2. Training (choose one model)

#### Their Model
```
python src/train.py --train-data-path data/data --name ours --causal-encoder --enc-attn-window -1 --aligned-cross-attn --requires-alignment --num-encoder-layers 10 --num-decoder-layers 2 --wait-k-cross-attn -1
```

#### Wait-k Model
```
python src/train.py --train-data-path data/data --name wait-3 --causal-encoder --enc-attn-window -1 --no-aligned-cross-attn --no-requires-alignment --num-encoder-layers 10 --num-decoder-layers 2 --wait-k-cross-attn 4 --no-modified-wait-k
```

#### Modified Wait-k Model
```
python src/train.py --train-data-path data/data --name wait-3-modified --causal-encoder --enc-attn-window -1 --no-aligned-cross-attn --no-requires-alignment --num-encoder-layers 10 --num-decoder-layers 2 --wait-k-cross-attn 4 --modified-wait-k
```

#### Retranslation Model
```
python src/train.py --train-data-path data/data --name retranslation --no-causal-encoder --enc-attn-window -1 --no-aligned-cross-attn --no-requires-alignment --num-encoder-layers 10 --num-decoder-layers 2 --wait-k-cross-attn -1 --no-modified-wait-k
```

### 3. Evaluation

#### Generate Predictions
```
python eval/evaluator.py --policy ours --test-data-path data/data --model-path logs/ours/checkpoints/last.ckpt --hparam-path logs/ours/hparams.yaml --output-path logs/ours/eval.pkl
```

#### Unpickle Results
```
python utils/unpickler.py --policy ours --pkl-path logs/ours/eval.pkl --train-vocab-path data/vocabs/train_kanji.json --test-vocab-path data/vocabs/train_kanji.json
```

### 4. Evaluation Metrics

#### Conversion Quality
```
python utils/score.py --prediction-path preds.txt --label-path labels.txt
```

#### Computational Latency
```
python utils/latency-c.py logs/ours/eval.pkl 
```

#### Non-computational Latency
```
python utils/latency.py logs/ours/eval.pkl --policy ours --kanji-vocab-path data/vocabs/train_kanji.json --kana-vocab-path data/vocabs/kana.json --label-path labels.txt
```
