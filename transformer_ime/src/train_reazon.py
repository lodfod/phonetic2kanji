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
import torch
from japanese_tokenizer import get_japanese_tokenizer
import wandb
import torch.distributed as dist

def setup_distributed_training():
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

def main():
    # Setup distributive training
    setup_distributed_training()

    # Initialize wandb
    wandb.init(project="kana-to-kanji", name="medium-training")
   
    parser = argparse.ArgumentParser(description="Train a kana to kanji model using Transformers")
    # Dataset and model arguments
    parser.add_argument("--dataset_dir", required=True, help="Directory with the formatted dataset")
    parser.add_argument("--model_name", default="google-t5/t5-small", 
                        help="Base model to fine-tune. Options include: google-t5/t5-small, google-t5/t5-base, " 
                             "rinna/japanese-gpt2-medium, cl-tohoku/bert-base-japanese")
    parser.add_argument("--output_dir", required=True, help="Directory to save the model")
    parser.add_argument("--tokenizer_type", default="mecab", choices=["mecab", "character"], 
                        help="Type of Japanese tokenizer to use")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Training hyperparameters
    # parser.add_argument("--num_epochs", type=int, default=10,
    #                     help="Number of training epochs")
    # parser.add_argument("--batch_size", type=int, default=64,
    #                     help="Batch size for training and evaluation")
    # parser.add_argument("--learning_rate", type=float, default=2e-5,
    #                     help="Learning rate")
    # parser.add_argument("--weight_decay", type=float, default=0.01,
    #                     help="Weight decay for AdamW optimizer")
    # parser.add_argument("--warmup_steps", type=int, default=500,
    #                     help="Number of warmup steps for learning rate scheduler")
    # parser.add_argument("--save_steps", type=int, default=400,
    #                     help="Number of steps between model saves")
    # parser.add_argument("--eval_steps", type=int, default=400,
    #                     help="Number of steps between evaluations")
    # parser.add_argument("--logging_steps", type=int, default=400,
    #                     help="Number of steps between logging")
    # parser.add_argument("--save_total_limit", type=int, default=3,
    #                     help="Maximum number of checkpoints to keep")
    
    
    args = parser.parse_args()
   
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
   
    # Load a Japanese-specific tokenizer
    tokenizer = get_japanese_tokenizer(args.tokenizer_type)
   
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
        evaluation_strategy="epoch",   # Evaluate every epoch
        save_strategy="epoch",   # Save every epoch
        learning_rate=2e-5,
        per_device_train_batch_size=256,   # Maximize GPU usage
        per_device_eval_batch_size=256,   # Maximize GPU usage
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,  # Test this out
        warmup_steps=500,
        fp16=True,
        group_by_length=True,
        report_to=["wandb", "tensorboard"],
        metric_for_best_model="cer",
        predict_with_generate=True,   # Test this out, most likely we need
        greater_is_better=False,   # Save model with smallest cer at end
        load_best_model_at_end=True,
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