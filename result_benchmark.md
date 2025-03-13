# Benchmark Results 

## General-context fine-tune performances

### mt5_small_medium 
#### Evaluated on jsut_basic
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
#### Evaluated on common_voice
```bash
================================================================================
Overall Results:
Precision: 69.7101
Recall: 69.2412
F-score: 69.2873
Character Error Rate (CER): 38.5703
Sentence Accuracy: 5.0146
================================================================================
```
#### Evaluated on reazon_tiny
```bash
================================================================================
Overall Results:
Precision: 77.8682
Recall: 77.3411
F-score: 77.4453
Character Error Rate (CER): 28.3915
Sentence Accuracy: 19.3493
================================================================================
```

### mt5_base_medium 
#### Evaluated on jsut_basic
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
#### Evaluated on common_voice
```bash
================================================================================
Overall Results:
Precision: 88.4472
Recall: 88.0769
F-score: 88.1963
Character Error Rate (CER): 13.7846
Sentence Accuracy: 31.0322
================================================================================
```
#### Evaluated on reazon_tiny
```bash
================================================================================
Overall Results:
Precision: 93.1510
Recall: 92.8777
F-score: 92.9734
Character Error Rate (CER): 8.3498
Sentence Accuracy: 50.5646
================================================================================
```

### mt5_small_all 
#### Evaluated on jsut_basic
```bash
================================================================================
Overall Results:
Precision: 89.9373
Recall: 90.1549
F-score: 90.0021
Character Error Rate (CER): 11.5602
Sentence Accuracy: 27.1864
================================================================================
```
#### Evaluated on common_voice
```bash
================================================================================
Overall Results:
Precision: 92.3169
Recall: 91.9361
F-score: 92.0763
Character Error Rate (CER): 9.2512
Sentence Accuracy: 43.0178
================================================================================
```
#### Evaluated on reazon_tiny
```bash
================================================================================
Overall Results:
Precision: 95.9574
Recall: 95.7327
F-score: 95.8204
Character Error Rate (CER): 4.8840
Sentence Accuracy: 64.2105
================================================================================
```

### mt5_base_all 
#### Evaluated on jsut_basic
```bash
================================================================================
Overall Results:
Precision: 92.9694
Recall: 93.0707
F-score: 92.9865
Character Error Rate (CER): 8.1646
Sentence Accuracy: 38.2549
================================================================================
```
#### Evaluated on common_voice
```bash
================================================================================
Overall Results:
Precision: 94.1808
Recall: 93.7334
F-score: 93.9156
Character Error Rate (CER): 7.1201
Sentence Accuracy: 50.8658
================================================================================
```
#### Evaluated on reazon_tiny
```bash
================================================================================
Overall Results:
Precision: 97.4678
Recall: 97.3430
F-score: 97.3863
Character Error Rate (CER): 3.0969
Sentence Accuracy: 75.0239
================================================================================
```
## Domain-specific fine-tune performances
### Evaluating models/wiki_technology_epoch_4/final_model on data/wiki_tech/test.kana and data/wiki_tech/test.kanji dataset
```bash
================================================================================
Overall Results:
Precision: 94.7934
Recall: 94.4910
F-score: 94.6089
Character Error Rate (CER): 6.5217
Sentence Accuracy: 38.9658
================================================================================
```
### Evaluating models/wiki_technology_epoch_6/final_model on data/wiki_tech/test.kana and data/wiki_tech/test.kanji dataset
```bash
================================================================================
Overall Results:
Precision: 95.2409
Recall: 94.9828
F-score: 95.0765
Character Error Rate (CER): 5.9428
Sentence Accuracy: 41.7334
================================================================================
```
### Evaluating models/wiki_technology_epoch_8/final_model on data/wiki_tech/test.kana and data/wiki_tech/test.kanji dataset
```bash
================================================================================
Overall Results:
Precision: 95.5528
Recall: 95.2879
F-score: 95.3886
Character Error Rate (CER): 5.5190
Sentence Accuracy: 43.4086
================================================================================
```
### Evaluating models/wiki_technology_epoch_10/final_model on data/wiki_tech/test.kana and data/wiki_tech/test.kanji dataset
```bash
================================================================================
Overall Results:
Precision: 95.7715
Recall: 95.4764
F-score: 95.5926
Character Error Rate (CER): 5.3338
Sentence Accuracy: 44.2098
================================================================================
```
### Evaluating models/mt5_base_all/final_model on data/wiki_tech/test.kana and data/wiki_tech/test.kanji dataset
```bash
================================================================================
Overall Results:
Precision: 92.1561
Recall: 92.0782
F-score: 92.0742
Character Error Rate (CER): 9.5157
Sentence Accuracy: 27.3853
================================================================================
```
## Effects of domain-specific fine tuning on general-context performance
### Evaluating models/wiki_technology_epoch_4/final_model on data/common_voice/clean_data.kana and data/common_voice/clean_data.kanji dataset
```bash
================================================================================
Overall Results:
Precision: 94.1635
Recall: 93.7486
F-score: 93.9123
Character Error Rate (CER): 7.1307
Sentence Accuracy: 51.0007
================================================================================
```
### Evaluating models/wiki_technology_epoch_6/final_model on data/common_voice/clean_data.kana and data/common_voice/clean_data.kanji dataset
```bash
================================================================================
Overall Results:
Precision: 94.1561
Recall: 93.7318
F-score: 93.9000
Character Error Rate (CER): 7.1466
Sentence Accuracy: 51.1356
================================================================================
```
### Evaluating models/wiki_technology_epoch_8/final_model on data/common_voice/clean_data.kana and data/common_voice/clean_data.kanji dataset
```bash
================================================================================
Overall Results:
Precision: 94.1400
Recall: 93.7164
F-score: 93.8848
Character Error Rate (CER): 7.1518
Sentence Accuracy: 51.2255
================================================================================
```
### Evaluating models/wiki_technology_epoch_10/final_model on data/common_voice/clean_data.kana and data/common_voice/clean_data.kanji dataset
```bash
================================================================================
Overall Results:
Precision: 94.1149
Recall: 93.6920
F-score: 93.8598
Character Error Rate (CER): 7.1838
Sentence Accuracy: 50.9782
================================================================================
```
## Default Non Neural IME
### Evaluated on jsut_basic
```bash
Overall Results:
================================================================================
Precision: 77.1769
Recall: 80.5462
F-score: 78.7320
Character Error Rate (CER): 27.8395
Sentence Accuracy: 5.4130
================================================================================
```
### Evaluated on common_voice
```bash
================================================================================
Precision: 80.4461
Recall: 83.0547
F-score: 81.6168
Character Error Rate (CER): 23.3107
Sentence Accuracy: 14.4592
================================================================================
```
### Evaluated on reazon_tiny
```bash
Overall Results:
================================================================================
Precision: 79.1907
Recall: 82.0659
F-score: 80.4663
Character Error Rate (CER): 25.1117
Sentence Accuracy: 19.7512
================================================================================
```