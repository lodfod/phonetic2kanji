import argparse
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import numpy as np
from datasets import load_from_disk

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a kana to kanji model on Wikipedia data")
    parser.add_argument("--base_model_dir", required=True, help="Directory with the base model")
    parser.add_argument("--dataset_dir", required=True, help="Directory with the Wikipedia formatted dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Load dataset
    wiki_dataset = load_from_disk(args.dataset_dir)
    
    # Load tokenizer and model from the base model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_dir)
    
    # Define the preprocessing function
    def preprocess_function(examples):
        inputs = ["translate kana to kanji: " + kana for kana in examples["translation"]["source"]]
        targets = examples["translation"]["target"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Preprocess the dataset
    tokenized_dataset = wiki_dataset.map(preprocess_function, batched=True)
    
    # Load metric
    metric = evaluate.load("sacrebleu")
    
    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Decode predictions
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode labels
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute BLEU score
        result = metric.compute(predictions=preds, references=[[l] for l in labels])
        
        # Add mean generated length
        result = {"bleu": result["score"]}
        
        return result
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        fp16=True,
        report_to="tensorboard",
    )
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        padding=True
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    trainer.train()
    
    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    
    # Save the tokenizer
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    print(f"Fine-tuned model saved to {os.path.join(args.output_dir, 'final_model')}")

if __name__ == "__main__":
    main() 