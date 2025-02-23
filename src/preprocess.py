# Kanji-to-Phonetic Conversion:

# Use MeCab to convert Kanji subtitle data into Hiragana.
# Write a Python script that calls the MeCab parser on each sentence.
# Validate outputs by checking for edge cases (e.g., ambiguous tokenization).
# Implementation Tip: Use the mecab-python3 package for easy integration with Python.
# Data Cleaning:

# Remove or correct any mismatches (e.g., extra spaces, punctuation inconsistencies).
# Create logging to monitor conversion accuracy.

import MeCab
import re

def convert_kanji_to_hiragana(text):
    # Initialize MeCab
    tagger = MeCab.Tagger("-Ochasen")
    
    # Parse the text
    parsed_text = tagger.parse(text)
    
    # Extract the hiragana
    hiragana = parsed_text.split("\t")[1]
    
    return hiragana

def clean_text(text):
    # Remove extra spaces and punctuation
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_text(text):
    # Convert to hiragana
    hiragana = convert_kanji_to_hiragana(text)
    
    # Clean the text
    cleaned_text = clean_text(hiragana)
    
    return cleaned_text

def preprocess_dataset(dataset):
    # Apply preprocessing to each sentence in the dataset
    processed_dataset = dataset.map(lambda x: {"sentence": preprocess_text(x["sentence"])})
    
    return processed_dataset


