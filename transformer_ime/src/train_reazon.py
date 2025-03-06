import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import numpy as np
from datasets import load_from_disk
import torch
import wandb

def setup_training_environment():
    # Disable distributed training
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU only
    # Clear any existing distributed training settings
    for var in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
        if var in os.environ:
            del os.environ[var]

def main():
    # Setup single GPU training
    setup_training_environment()
   
    parser = argparse.ArgumentParser(description="Train a kana to kanji model using Transformers")
    # Dataset and model arguments
    parser.add_argument("--dataset_dir", required=True, help="Directory with the formatted dataset")
    parser.add_argument("--model_name", default="google-t5/t5-small", 
                        help="Base model to fine-tune. Options include: google-t5/t5-small, google-t5/t5-base, " 
                             "rinna/japanese-gpt2-medium, cl-tohoku/bert-base-japanese, google/mt5-small, google/mt5-base")
    parser.add_argument("--output_dir", required=True, help="Directory to save the model")
    parser.add_argument("--tokenizer_type", default="mecab", choices=["mecab", "character"], 
                        help="Type of Japanese tokenizer to use")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")  
    args = parser.parse_args()
    
    # Initialize wandb with run name from output_dir
    run_name = os.path.basename(args.output_dir)
    wandb.init(project="kana-to-kanji", name=run_name)

    # Set up device for training
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for accelerated training")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for accelerated training")
    else:
        print("MPS device not found, using CPU")
   
    # Load dataset
    dataset = load_from_disk(args.dataset_dir)
    print("Dataset structure:", dataset)
    print("First example:", dataset["train"][0] if "train" in dataset else dataset[0])
   
    # Load default tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
   
    # Define the preprocessing function
    def preprocess_function(examples):
        # print("Type of examples:", type(examples))
        # print("Keys in examples:", list(examples.keys()))
        # print("Type of translation:", type(examples["translation"]))
       
        # if isinstance(examples["translation"], dict):
        #     print("Keys in translation:", list(examples["translation"].keys()))
       
        # When batched=True, examples["translation"] might be a list of dictionaries
        # or a dictionary with lists as values, depending on the dataset structure
        if isinstance(examples["translation"], list):
            # Handle case where translation is a list of dictionaries
            inputs = ["translate kana to kanji: " + item["source"] for item in examples["translation"]]
            targets = [item["target"] for item in examples["translation"]]
        else:
            # Handle case where translation is a dictionary with lists
            inputs = ["translate kana to kanji: " + kana for kana in examples["translation"]["source"]]
            targets = examples["translation"]["target"]
       
        model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True)
       
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_length, truncation=True)
           
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
   
    # Preprocess the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
   
    # Load CER metric only
    cer_metric = evaluate.load("cer")
   
    # Define compute metrics function for CER only
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Decode predictions
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode labels
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Filter out empty strings
        filtered_pairs = [(p, r) for p, r in zip(preds, labels) if len(r.strip()) > 0]
        if not filtered_pairs:
            return {"cer": 1.0}  # Return worst score if all references are empty
        filtered_preds, filtered_labels = zip(*filtered_pairs)
        
        # Compute CER score
        cer_score = cer_metric.compute(predictions=filtered_preds, references=filtered_labels)
        
        return {"cer": cer_score}
   
    # Initialize model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
   
    # Define training arguments using parsed arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",         # Evaluate every epoch
        save_strategy="epoch",               # Save every epoch
        learning_rate=2e-5,                  # TEST
        per_device_train_batch_size=256,     # Maximize GPU usage
        per_device_eval_batch_size=256,      # Maximize GPU usage
        auto_find_batch_size=True,           # Automatically lower batch size to prevent memory error
        weight_decay=0.01,                   # L2 regularization to prevent overfitting
        save_total_limit=3,                  # Save up to three checkpoints
        num_train_epochs=10,                 # TEST
        warmup_steps=500,                    # Help stabilize early training
        fp16=True,                           # Reduces memory usage and speeds up training
        group_by_length=True,                # Removes unnecessary padding
        report_to=["wandb", "tensorboard"],  # Report to wandb and tensorboard
        metric_for_best_model="cer",         # Evaluate with CER
        predict_with_generate=True,          # Test this out, most likely we need
        greater_is_better=False,             # Save model with smallest cer at end
        load_best_model_at_end=True,         # Save best model
        max_grad_norm=1.0,                   # Added gradient clipping
        local_rank=-1,                       # Disable distributed training for single GPU training
        ddp_backend=None,                    # Disable DDP for single GPU training
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
   
    # Update wandb config to use all arguments
    wandb.config.update(vars(args))
   
    # Train model
    trainer.train()
   
    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
   
    # Save the tokenizer
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
   
    # Close wandb run
    wandb.finish()
   
    print(f"Model saved to {os.path.join(args.output_dir, 'final_model')}")


if __name__ == "__main__":
    main()